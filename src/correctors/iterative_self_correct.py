"""方向4：迭代自纠正循环。

多轮生成→自检→重读→修正，追踪每轮的ISR/NAR变化。

流程：Round 0 初答 → 自检矛盾 → 标记可疑文档 →
      Round 1 重读+修订 → 再自检 → ... → 最多3轮或收敛则停

API调用：n_rounds × 2 (生成+自检) + 1(最终) = 3~7次
"""
from __future__ import annotations

import time

from ..noise_injector import NoisyContext
from ..prompts import format_context
from ..rag_pipeline import RAGResult
from ..metrics import attribute_answer, tokenize, token_set
from .base import BaseCorrector, register_corrector


# 自检 prompt
SELF_CHECK_SYSTEM = (
    "你是答案审阅员。给定问题、相关文档、和候选答案，"
    "请检查答案中的每条事实是否能在至少一篇文档中找到明确支撑。\n"
    "输出格式：\n"
    "一致性：PASS（所有事实有文档支撑）或 FAIL（存在无支撑或与文档矛盾的事实）\n"
    "问题描述：[若FAIL，简述哪条事实有问题]"
)

SELF_CHECK_USER = (
    "【问题】{query}\n\n"
    "【文档】\n{context}\n\n"
    "【候选答案】{answer}\n\n"
    "请检查一致性："
)

# 修订 prompt
REVISE_SYSTEM = (
    "你是严谨的答案修订员。之前的答案可能存在问题：{issue}"
    "请重新审阅所有文档，仅基于文档中的可靠信息给出修正后的答案。"
    "答案尽量简短（短语或一句话），不要解释。"
)

REVISE_USER = (
    "【问题】{query}\n\n"
    "【文档】\n{context}\n\n"
    "【初版答案】{prev_answer}\n\n"
    "请输出修正后的答案："
)


def _check_pass(content: str) -> bool:
    text = (content or "").strip().upper()
    return "PASS" in text and "FAIL" not in text


def _extract_issue(content: str) -> str:
    text = (content or "").strip()
    for line in text.split("\n"):
        if "问题描述" in line or "问题" in line:
            return line.split("：", 1)[-1].strip() if "：" in line else line.strip()
    return text[:200]


@register_corrector("iterative_sc")
class IterativeSelfCorrectCorrector(BaseCorrector):
    """迭代自纠正：最多3轮生成→自检→修订。"""

    api_cost = 0
    max_rounds: int = 3

    def correct(self, ctx: NoisyContext, *, language: str = "zh") -> RAGResult:
        docs_ctx = format_context(ctx.docs)
        api_calls = 0
        prompt_tokens = 0
        completion_tokens = 0
        round_log: list[dict] = []
        t0 = time.perf_counter()

        # Round 0: 初次生成
        gen = self.llm.chat(
            [
                {"role": "system", "content": "你是严谨的问答助手。请仅基于提供的文档作答，答案尽量简短。"},
                {"role": "user", "content": (
                    f"【问题】{ctx.query}\n\n【文档】共{len(ctx.docs)}篇：\n{docs_ctx}\n\n请简短作答："
                )},
            ],
        )
        api_calls += 1
        prompt_tokens += gen.get("prompt_tokens", 0)
        completion_tokens += gen.get("completion_tokens", 0)
        current_answer = (gen["content"] or "").strip()

        round_log.append({
            "round": 0,
            "answer": current_answer,
            "action": "initial_generation",
        })

        # 迭代轮次
        for rnd in range(1, self.max_rounds + 1):
            # 自检
            check = self.llm.chat(
                [
                    {"role": "system", "content": SELF_CHECK_SYSTEM},
                    {"role": "user", "content": SELF_CHECK_USER.format(
                        query=ctx.query,
                        context=docs_ctx,
                        answer=current_answer,
                    )},
                ],
                max_tokens=150,
            )
            api_calls += 1
            prompt_tokens += check.get("prompt_tokens", 0)
            completion_tokens += check.get("completion_tokens", 0)

            check_content = check.get("content", "") or ""

            if _check_pass(check_content):
                round_log.append({
                    "round": rnd,
                    "answer": current_answer,
                    "action": "pass",
                    "check_output": check_content[:200],
                })
                break  # 收敛，停止迭代

            issue = _extract_issue(check_content)

            # 修订
            revise = self.llm.chat(
                [
                    {"role": "system", "content": REVISE_SYSTEM.format(issue=issue)},
                    {"role": "user", "content": REVISE_USER.format(
                        query=ctx.query,
                        context=docs_ctx,
                        prev_answer=current_answer,
                    )},
                ],
            )
            api_calls += 1
            prompt_tokens += revise.get("prompt_tokens", 0)
            completion_tokens += revise.get("completion_tokens", 0)

            new_answer = (revise["content"] or "").strip() if revise.get("content") else current_answer

            round_log.append({
                "round": rnd,
                "answer": new_answer,
                "prev_answer": current_answer,
                "action": "revised",
                "issue": issue[:200],
                "check_output": check_content[:200],
            })
            current_answer = new_answer

        elapsed = time.perf_counter() - t0

        return RAGResult(
            sample_id=ctx.sample_id,
            query=ctx.query,
            gold_answers=ctx.gold_answers,
            prediction=current_answer,
            docs=ctx.docs,
            labels=list(ctx.labels),
            noise_ratio=ctx.noise_ratio,
            noise_type=ctx.noise_type,
            noise_position=ctx.noise_position,
            metadata={
                "method": self.name,
                "api_calls": api_calls,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency": elapsed,
                "rounds": len(round_log),
                "round_log": round_log,
                "converged": len(round_log) < self.max_rounds + 1,
                **ctx.meta,
            },
        )
