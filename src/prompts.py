"""所有 prompt 模板集中管理 —— 修改一处即可全局生效。

约定：
- {query}: 用户问题
- {context}: 已拼接好的文档块 (含编号)
- {n}: 文档总数
"""
from __future__ import annotations

NAIVE_SYSTEM_ZH = (
    "你是一个严谨的问答助手。给定若干检索文档和一个问题，"
    "请仅基于这些文档作答，答案务必简洁。"
    "如果文档中找不到答案，请回答“无法回答”。"
)
NAIVE_SYSTEM_EN = (
    "You are a rigorous QA assistant. Given several retrieved documents and a question, "
    "answer concisely based ONLY on the documents. "
    'If the answer is not found, reply "I cannot answer".'
)

NAIVE_USER_TMPL = (
    "【检索文档】共{n}篇：\n"
    "{context}\n\n"
    "【问题】{query}\n\n"
    "【要求】答案尽量简短（短语或一句话），不要解释。"
)


PROMPT_AWARE_SYSTEM_ZH = (
    "你是一个具备质疑能力的问答助手。检索结果中可能掺杂【与问题语义相关但不含答案】"
    "或【信息错误】的噪音文档。请：\n"
    "1. 先识别每篇文档是否对回答该问题真正有用；\n"
    "2. 只基于真正有用的文档作答；\n"
    "3. 若所有文档都不能支撑答案，回答“无法回答”。"
)

PROMPT_AWARE_SYSTEM_EN = (
    "You are a critical QA assistant. The retrieved documents may contain noise: "
    "semantically related but answer-irrelevant documents, or factually incorrect ones. "
    "Please: 1) Judge whether each document truly helps answer the question; "
    "2) Answer based ONLY on genuinely useful documents; "
    '3) If no document supports an answer, reply "I cannot answer".'
)



COT_EVIDENCE_SYSTEM_ZH = (
    "你是一个证据链推理助手。请严格遵循以下步骤：\n"
    "Step1. 拆解问题：列出回答该问题需要哪些关键信息点；\n"
    "Step2. 逐一检查文档：标注哪些文档提供了哪些信息点；\n"
    "Step3. 综合证据：仅基于已找到证据的信息点构建答案；\n"
    "Step4. 输出最终答案（一行，简短）。\n\n"
    "请用如下结构输出：\n"
    "信息需求：...\n"
    "证据匹配：文档[i]→信息点x；...\n"
    "最终答案：<answer>...</answer>"
)


COT_EVIDENCE_SYSTEM_EN = (
    "You are an evidence-chain reasoning assistant. Follow these steps strictly:\n"
    "Step1. Decompose the question: list the key information points needed;\n"
    "Step2. Check each document: note which documents provide which points;\n"
    "Step3. Synthesize: build the answer ONLY from evidenced points;\n"
    "Step4. Output the final answer (one line, concise).\n\n"
    "Use this structure:\n"
    "Information needs: ...\n"
    "Evidence mapping: Doc[i]→point x; ...\n"
    "Final answer: <answer>...</answer>"
)


ITER_FILTER_SYSTEM_ZH = (
    "你是文档相关性评估器。对于给定的问题和单篇文档，"
    "请只输出 high / mid / low 三个标签之一：\n"
    "- high：文档直接包含答案；\n"
    "- mid：文档与问题主题相关但不含答案；\n"
    "- low：文档与问题无关，或信息明显错误。"
)

ITER_FILTER_USER_TMPL = (
    "【问题】{query}\n\n【文档】{doc}\n\n请输出标签（high/mid/low）："
)

ITER_FILTER_SYSTEM_EN = (
    "You are a document relevance assessor. For a given question and a single document, "
    "output ONLY one of three labels:\n"
    "- high: the document directly contains the answer;\n"
    "- mid: the document is topically related but does not contain the answer;\n"
    "- low: the document is irrelevant or contains clearly incorrect information."
)
ITER_FILTER_USER_TMPL_EN = (
    "Question: {query}\n\nDocument: {doc}\n\nOutput label (high/mid/low):"
)


SELFRAG_REL_SYSTEM_ZH = (
    "判断给定文档对回答给定问题是否相关。只输出 RELEVANT 或 IRRELEVANT。"
)
SELFRAG_REL_USER_TMPL = "【问题】{query}\n\n【文档】{doc}\n\n判断："

SELFRAG_REL_SYSTEM_EN = (
    "Determine if the given document is relevant to answering the given question. "
    "Output ONLY RELEVANT or IRRELEVANT."
)
SELFRAG_REL_USER_TMPL_EN = "Question: {query}\n\nDocument: {doc}\n\nJudgment:"

SELFRAG_SUPPORT_SYSTEM_ZH = (
    "判断给定答案是否得到给定文档的支撑。只输出 SUPPORTED / PARTIAL / UNSUPPORTED。"
)
SELFRAG_SUPPORT_USER_TMPL = "【问题】{query}\n\n【答案】{answer}\n\n【相关文档】\n{context}\n\n判断："

SELFRAG_SUPPORT_SYSTEM_EN = (
    "Determine if the given answer is supported by the given documents. "
    "Output ONLY SUPPORTED / PARTIAL / UNSUPPORTED."
)
SELFRAG_SUPPORT_USER_TMPL_EN = "Question: {query}\n\nAnswer: {answer}\n\nRelevant documents:\n{context}\n\nJudgment:"


JUDGE_SYSTEM_ZH = (
    "你是 QA 答案评分员。给定问题、参考答案、模型答案，请按 1-5 整数打分：\n"
    "5=语义完全正确；4=基本正确；3=部分正确；2=有相关但关键错误；1=完全错误。\n"
    "只输出一个整数。"
)
JUDGE_USER_TMPL = (
    "【问题】{query}\n【参考答案】{gold}\n【模型答案】{pred}\n请打分(1-5)："
)


# ===== 方法 D · 证据投票 (Evidence Voting) =====

VOTE_PROMPTS_ZH: tuple[str, ...] = (
    # Persona 1: 严谨的事实核查员
    (
        "你是严谨的事实核查员。请仅根据提供的文档作答。"
        "对每条事实必须能从文档中找到出处，否则回答“无法回答”。简短作答。"
    ),
    # Persona 2: 多疑的研究员
    (
        "你是多疑的研究员。检索文档中可能包含语义相关但错误或与问题无关的信息。"
        "请识别可信文档，仅基于可信信息作答。简短作答。"
    ),
    # Persona 3: 证据链推理者
    (
        "你是证据链推理者。请先在内心思考：问题需要哪些关键信息？哪些文档真正提供了？"
        "然后只输出最终答案（一行，简短）。"
    ),
)


VOTE_AGGREGATE_SYSTEM_ZH = (
    "你是答案聚合器。给定同一问题的 3 个候选答案，请输出最可靠、最简洁的最终答案。"
    "若候选答案矛盾，请基于多数原则与事实合理性选择；若全部都不可靠，回答“无法回答”。"
    "只输出最终答案，不要解释。"
)

VOTE_AGGREGATE_USER_TMPL = (
    "【问题】{query}\n"
    "【候选答案1】{cand1}\n"
    "【候选答案2】{cand2}\n"
    "【候选答案3】{cand3}\n"
    "请输出最终答案（一行，简短）："
)


def format_context(docs: list[str], *, max_chars_per_doc: int = 1500) -> str:
    """把若干文档拼成带编号的上下文块，并裁剪过长文档。"""
    parts: list[str] = []
    for i, d in enumerate(docs):
        text = d.strip()
        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc] + "..."
        parts.append(f"[文档{i}] {text}")
    return "\n\n".join(parts)


def build_naive_prompt(query: str, docs: list[str], *, language: str = "zh") -> tuple[str, str]:
    sys_msg = NAIVE_SYSTEM_ZH if language == "zh" else NAIVE_SYSTEM_EN
    user_msg = NAIVE_USER_TMPL.format(query=query, n=len(docs), context=format_context(docs))
    return sys_msg, user_msg
