"""所有 prompt 模板集中管理 —— 修改一处即可全局生效。

约定：
- {query}: 用户问题
- {context}: 已拼接好的文档块 (含编号)
- {n}: 文档总数
"""
from __future__ import annotations

from dataclasses import dataclass

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

NAIVE_USER_TMPL_EN = (
    "Retrieved documents ({n}):\n"
    "{context}\n\n"
    "Question: {query}\n\n"
    "Answer in a short phrase or one sentence. No explanation."
)

# ── CmedqaRetrieval · 中文医学咨询 ──
NAIVE_CMEDQA_SYSTEM_ZH = (
    "你是面向患者的医学健康咨询助手。给定若干检索到的医生回复/参考资料和一个患者问题，"
    "请仅基于这些文档作答，语气与在线医生答疑相近。"
    "回答需说明可能原因或机制、具体处理/生活建议，必要时提示何时就医；"
    "不要只输出单个诊断词、元素名称或一句话标签。"
    "若文档无法支撑回答，请回答“无法回答”；不要编造文档中未出现的检查或药物。"
)
NAIVE_CMEDQA_USER_TMPL = (
    "【参考资料】共{n}篇：\n"
    "{context}\n\n"
    "【患者问题】{query}\n\n"
    "【作答要求】请用完整中文段落作答（通常 2–5 句）："
    "先结合文档解释可能原因，再给出可操作建议；可综合多篇资料，但不要只写一句话。"
)

# ── MIRIAD · 英文医学 ──
NAIVE_MIRIAD_SYSTEM_EN = (
    "You are a clinical health QA assistant. Given retrieved medical passages and a patient question, "
    "answer ONLY from the documents in a clinician-advising tone. "
    "Explain likely cause/mechanism, practical recommendations, and when to seek care when relevant. "
    "Do not reply with a one-line label, single nutrient, or diagnosis name only. "
    'If unsupported, reply "I cannot answer". Do not invent tests or drugs not in the documents.'
)
NAIVE_MIRIAD_USER_TMPL_EN = (
    "Reference passages ({n}):\n"
    "{context}\n\n"
    "Patient question: {query}\n\n"
    "Answer in a complete paragraph (typically 2–5 sentences) with rationale and advice."
)

# ── 2WikiMultihopQA · 英文事实/多跳 ──
NAIVE_2WIKI_SYSTEM_EN = (
    "You are a rigorous QA assistant for multi-hop factual questions. "
    "Given retrieved Wikipedia-style passages, answer ONLY from the documents. "
    "Output the required entity, name, date, place, or yes/no—keep it short (phrase or one sentence). "
    'If the answer is not found, reply "I cannot answer".'
)
NAIVE_2WIKI_USER_TMPL_EN = (
    "Retrieved documents ({n}):\n"
    "{context}\n\n"
    "Question: {query}\n\n"
    "Answer with the final fact only (short phrase or one sentence). No explanation."
)

# ── BRIGHT · long-form reasoning QA ──
NAIVE_BRIGHT_SYSTEM_EN = (
    "You are a rigorous QA assistant for long-form reasoning questions. "
    "Given retrieved passages, answer ONLY from the documents with clear reasoning. "
    "Write a complete paragraph explaining the answer; do not reply with a single phrase only. "
    'If unsupported, reply "I cannot answer".'
)
NAIVE_BRIGHT_USER_TMPL_EN = (
    "Retrieved documents ({n}):\n"
    "{context}\n\n"
    "Question: {query}\n\n"
    "Answer in a complete paragraph with reasoning grounded in the documents."
)

# ── MultiHop-RAG · news multi-hop ──
NAIVE_MULTIHOP_RAG_SYSTEM_EN = (
    "You are a rigorous QA assistant for multi-hop news questions. "
    "Given retrieved news passages, answer ONLY from the documents. "
    "Output the required entity, name, date, or short fact—keep it concise. "
    'If the answer is not found, reply "I cannot answer".'
)
NAIVE_MULTIHOP_RAG_USER_TMPL_EN = (
    "Retrieved documents ({n}):\n"
    "{context}\n\n"
    "Question: {query}\n\n"
    "Answer with the final fact only (short phrase or one sentence). No explanation."
)

# ── TEMPO · forum long-form QA ──
NAIVE_TEMPO_SYSTEM_EN = (
    "You are a forum-style QA assistant. Given retrieved discussion passages and a question, "
    "answer ONLY from the documents in a helpful, explanatory tone. "
    "Write a complete paragraph with practical details when relevant. "
    'If unsupported, reply "I cannot answer".'
)
NAIVE_TEMPO_USER_TMPL_EN = (
    "Retrieved documents ({n}):\n"
    "{context}\n\n"
    "Question: {query}\n\n"
    "Answer in a complete paragraph (typically 2–5 sentences) grounded in the documents."
)


@dataclass(frozen=True)
class NaivePromptProfile:
    """按数据集区分的 naive RAG 提示词 profile。"""

    system_zh: str
    system_en: str
    user_zh: str
    user_en: str

    def system(self, *, language: str) -> str:
        return self.system_zh if language == "zh" else self.system_en

    def user_tmpl(self, *, language: str) -> str:
        return self.user_zh if language == "zh" else self.user_en


NAIVE_PROFILES: dict[str, NaivePromptProfile] = {
    "rgb": NaivePromptProfile(
        system_zh=NAIVE_SYSTEM_ZH,
        system_en=NAIVE_SYSTEM_EN,
        user_zh=NAIVE_USER_TMPL,
        user_en=NAIVE_USER_TMPL_EN,
    ),
    "cmedqa": NaivePromptProfile(
        system_zh=NAIVE_CMEDQA_SYSTEM_ZH,
        system_en=NAIVE_SYSTEM_EN,
        user_zh=NAIVE_CMEDQA_USER_TMPL,
        user_en=NAIVE_USER_TMPL_EN,
    ),
    "miriad": NaivePromptProfile(
        system_zh=NAIVE_SYSTEM_ZH,
        system_en=NAIVE_MIRIAD_SYSTEM_EN,
        user_zh=NAIVE_USER_TMPL,
        user_en=NAIVE_MIRIAD_USER_TMPL_EN,
    ),
    "2wiki": NaivePromptProfile(
        system_zh=NAIVE_SYSTEM_ZH,
        system_en=NAIVE_2WIKI_SYSTEM_EN,
        user_zh=NAIVE_USER_TMPL,
        user_en=NAIVE_2WIKI_USER_TMPL_EN,
    ),
    "bright": NaivePromptProfile(
        system_zh=NAIVE_SYSTEM_ZH,
        system_en=NAIVE_BRIGHT_SYSTEM_EN,
        user_zh=NAIVE_USER_TMPL,
        user_en=NAIVE_BRIGHT_USER_TMPL_EN,
    ),
    "multihop_rag": NaivePromptProfile(
        system_zh=NAIVE_SYSTEM_ZH,
        system_en=NAIVE_MULTIHOP_RAG_SYSTEM_EN,
        user_zh=NAIVE_USER_TMPL,
        user_en=NAIVE_MULTIHOP_RAG_USER_TMPL_EN,
    ),
    "tempo": NaivePromptProfile(
        system_zh=NAIVE_SYSTEM_ZH,
        system_en=NAIVE_TEMPO_SYSTEM_EN,
        user_zh=NAIVE_USER_TMPL,
        user_en=NAIVE_TEMPO_USER_TMPL_EN,
    ),
}


def normalize_dataset_key(dataset: str | None) -> str:
    key = (dataset or "rgb").strip().lower()
    return key if key in NAIVE_PROFILES else "rgb"


def get_naive_profile(dataset: str | None) -> NaivePromptProfile:
    return NAIVE_PROFILES[normalize_dataset_key(dataset)]


def context_dataset(ctx) -> str:
    """从 NoisyContext.meta 读取数据集 id。"""
    meta = getattr(ctx, "meta", None) or {}
    return normalize_dataset_key(meta.get("dataset"))


LONG_FORM_DATASETS: frozenset[str] = frozenset({"bright", "tempo", "cmedqa", "miriad"})


def is_long_form_dataset(dataset: str | None) -> bool:
    return normalize_dataset_key(dataset) in LONG_FORM_DATASETS

CLOSED_BOOK_SYSTEM_ZH = (
    "你是一个严谨的问答助手。请仅根据你已有的知识回答问题，"
    "答案务必简洁。如果不确定，请回答“无法回答”。"
)
CLOSED_BOOK_SYSTEM_EN = (
    "You are a rigorous QA assistant. Answer concisely based ONLY on your own knowledge. "
    'If you are unsure, reply "I cannot answer".'
)
CLOSED_BOOK_USER_TMPL_ZH = "【问题】{query}\n\n【要求】答案尽量简短（短语或一句话），不要解释。"
CLOSED_BOOK_USER_TMPL_EN = "Question: {query}\n\nAnswer concisely in a phrase or one sentence. No explanation."


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
    "你是 QA 答案评判员。给定标准答案（label）和模型预测（prediction），"
    "判断预测是否在语义上与标准答案一致。\n"
    "硬性规则（优先级最高）：\n"
    "- 人数、日期、年份、比例、金额、停经天数、测量数值等必须与 label 一致；"
    "label 问总人数时，只答子集人数（如只答儿童数）视为错误。\n"
    "- 数值不同（如 19 vs 21）即使话题相关，最高只能给 2 分。\n"
    "- 预测仅给出最终结论/诊断名称，而 label 还包含依据、关键发现、测量或推理过程时，"
    "即使结论正确，最高只能给 3 分。\n"
    "- 医学/临床答案：label 中的超声所见、胚胎结构、数值等关键信息在 prediction 中缺失时，"
    "不得给 4 分及以上。\n"
    "评分标准（只输出一个整数）：\n"
    "5=语义完全正确且覆盖 label 关键信息；4=基本正确，仅有轻微表述差异；"
    "3=部分正确或仅结论正确但缺少关键依据；2=有关但关键事实错误；1=完全错误或无关。\n"
    "只输出一个 1-5 的整数，不要解释。"
)
JUDGE_USER_TMPL_ZH = (
    "【标准答案 label】{gold}\n"
    "【模型预测 prediction】{pred}\n"
    "【问题（可选参考）】{query}\n"
    "请打分(1-5)："
)

JUDGE_SYSTEM_EN = (
    "You are a QA answer evaluator. Given the reference label and the model prediction, "
    "judge whether the prediction is semantically correct.\n"
    "Hard rules (highest priority):\n"
    "- Counts, dates, years, ratios, amounts, gestational days, and measurements must match the label.\n"
    "- If the label asks for a total count, answering only a subset count is wrong.\n"
    "- Different numbers (e.g. 19 vs 21) may score at most 2 even if on-topic.\n"
    "- If the prediction gives only a final conclusion/diagnosis while the label also includes "
    "evidence, key findings, measurements, or reasoning, score at most 3 even if the conclusion matches.\n"
    "- Medical/clinical answers: missing key findings from the label (imaging, structures, values) "
    "cannot score 4 or 5.\n"
    "Scoring (output ONE integer only):\n"
    "5=fully correct and covers key label information; 4=mostly correct with minor wording differences; "
    "3=partially correct, or conclusion-only without key supporting details; "
    "2=related but key factual errors; 1=completely wrong or irrelevant.\n"
    "Output a single integer from 1 to 5. No explanation."
)
JUDGE_USER_TMPL_EN = (
    "Reference label: {gold}\n"
    "Model prediction: {pred}\n"
    "Question (optional context): {query}\n"
    "Score (1-5):"
)

# ── 数据集特化 Judge · 中文医学 ──
JUDGE_CMEDQA_SYSTEM_ZH = (
    "你是中文医学 QA 评判员。label 通常是医生给出的完整答疑段落（含原因、建议、就医提示）。"
    "prediction 也应对患者给出完整段落，而非单个诊断词。\n"
    "硬性规则：\n"
    "- 预测仅输出结论/诊断名称而 label 含超声所见、数值、处理建议时，最高 3 分。\n"
    "- 数值、停经天数、测量必须与 label 一致；数值错误最高 2 分。\n"
    "5=语义正确且覆盖 label 关键临床信息；4=基本正确；3=部分正确或仅结论正确；"
    "2=有关但关键事实错；1=完全错误。只输出 1-5 整数。"
)

# ── Judge · MIRIAD 英文医学 ──
JUDGE_MIRIAD_SYSTEM_EN = (
    "You are a clinical QA evaluator. Labels are full patient-advice paragraphs. "
    "Predictions should match in mechanism, recommendations, and care guidance—not a one-line label.\n"
    "Hard rules:\n"
    "- Conclusion-only prediction when label includes findings/advice: max 3.\n"
    "- Missing key clinical findings from label: cannot score 4 or 5.\n"
    "5=fully correct with key info; 4=mostly correct; 3=partial/conclusion-only; "
    "2=key errors; 1=wrong. Output ONE integer 1-5."
)

# ── Judge · 2Wiki / 短事实多跳 ──
JUDGE_2WIKI_SYSTEM_EN = (
    "You evaluate multi-hop factual QA. Labels are short: entity name, date, place, or yes/no.\n"
    "Hard rules:\n"
    "- The required entity/name/date must match the label (allow minor article/case differences).\n"
    "- Explanations are unnecessary; judge the final fact only.\n"
    "- Wrong entity or date: max 2 even if on-topic.\n"
    "5=exact/correct entity or fact; 4=minor wording diff; 3=partial; 2=wrong key fact; 1=irrelevant. "
    "Output ONE integer 1-5."
)

# ── Judge · BRIGHT 长文推理 ──
JUDGE_BRIGHT_SYSTEM_EN = (
    "You evaluate long-form reasoning QA (e.g. StackOverflow/science forums). "
    "Labels are full explanatory paragraphs with mechanisms and conclusions.\n"
    "Hard rules:\n"
    "- Score 5 if core conclusion AND main reasoning chain match the label semantically.\n"
    "- Score 4 if conclusion correct with minor missing reasoning detail.\n"
    "- Score 3 if directionally right but missing a key mechanism or evidence step.\n"
    "- A one-line phrase when the label is a full paragraph: max 2.\n"
    "- Ignore LaTeX/markdown/HTML formatting differences if ideas align.\n"
    "Output ONE integer 1-5. No explanation."
)

# ── Judge · MultiHop-RAG 新闻短答 ──
JUDGE_MULTIHOP_RAG_SYSTEM_EN = (
    "You evaluate multi-hop news QA. Labels are short: person name, company, date, or brief fact "
    "synthesized from multiple news passages.\n"
    "Hard rules:\n"
    "- Required name/entity/date must match the label.\n"
    "- Extra explanation is fine but judge the final fact; wrong person/entity: max 2.\n"
    "5=correct entity/fact; 4=minor wording; 3=partial; 2=wrong key fact; 1=irrelevant. "
    "Output ONE integer 1-5."
)

# ── Judge · TEMPO 论坛长文 ──
JUDGE_TEMPO_SYSTEM_EN = (
    "You evaluate forum-style QA. Labels may contain HTML and long practical answers "
    "(crypto, travel, workplace, etc.).\n"
    "Hard rules:\n"
    "- Judge semantic equivalence of advice, conclusions, and key practical details.\n"
    "- Ignore HTML tags and minor formatting; compare meaning.\n"
    "- Score 5 if prediction covers the label's main recommendation and rationale.\n"
    "- One-line tag when label is a full helpful paragraph: max 2.\n"
    "5=semantically equivalent and complete; 4=mostly correct; 3=partial; 2=wrong key point; "
    "1=irrelevant. Output ONE integer 1-5."
)


@dataclass(frozen=True)
class JudgePromptProfile:
    system_zh: str
    system_en: str
    user_zh: str
    user_en: str

    def system(self, *, language: str) -> str:
        return self.system_zh if language == "zh" else self.system_en

    def user_tmpl(self, *, language: str) -> str:
        return self.user_zh if language == "zh" else self.user_en


JUDGE_PROFILES: dict[str, JudgePromptProfile] = {
    "rgb": JudgePromptProfile(
        system_zh=JUDGE_SYSTEM_ZH,
        system_en=JUDGE_SYSTEM_EN,
        user_zh=JUDGE_USER_TMPL_ZH,
        user_en=JUDGE_USER_TMPL_EN,
    ),
    "cmedqa": JudgePromptProfile(
        system_zh=JUDGE_CMEDQA_SYSTEM_ZH,
        system_en=JUDGE_SYSTEM_EN,
        user_zh=JUDGE_USER_TMPL_ZH,
        user_en=JUDGE_USER_TMPL_EN,
    ),
    "miriad": JudgePromptProfile(
        system_zh=JUDGE_SYSTEM_ZH,
        system_en=JUDGE_MIRIAD_SYSTEM_EN,
        user_zh=JUDGE_USER_TMPL_ZH,
        user_en=JUDGE_USER_TMPL_EN,
    ),
    "2wiki": JudgePromptProfile(
        system_zh=JUDGE_SYSTEM_ZH,
        system_en=JUDGE_2WIKI_SYSTEM_EN,
        user_zh=JUDGE_USER_TMPL_ZH,
        user_en=JUDGE_USER_TMPL_EN,
    ),
    "bright": JudgePromptProfile(
        system_zh=JUDGE_SYSTEM_ZH,
        system_en=JUDGE_BRIGHT_SYSTEM_EN,
        user_zh=JUDGE_USER_TMPL_ZH,
        user_en=JUDGE_USER_TMPL_EN,
    ),
    "multihop_rag": JudgePromptProfile(
        system_zh=JUDGE_SYSTEM_ZH,
        system_en=JUDGE_MULTIHOP_RAG_SYSTEM_EN,
        user_zh=JUDGE_USER_TMPL_ZH,
        user_en=JUDGE_USER_TMPL_EN,
    ),
    "tempo": JudgePromptProfile(
        system_zh=JUDGE_SYSTEM_ZH,
        system_en=JUDGE_TEMPO_SYSTEM_EN,
        user_zh=JUDGE_USER_TMPL_ZH,
        user_en=JUDGE_USER_TMPL_EN,
    ),
}


def get_judge_profile(dataset: str | None) -> JudgePromptProfile:
    return JUDGE_PROFILES.get(normalize_dataset_key(dataset), JUDGE_PROFILES["rgb"])


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


# ===== Multihop decompose (2Wiki) =====

MULTIHOP_DECOMPOSE_SYSTEM_EN = (
    "You decompose multi-hop questions into 1-2 simpler sub-questions that must be "
    "answered IN ORDER. Each later sub-question may use the answer to the previous one.\n"
    "Rules:\n"
    "- Output ONLY a numbered list (1. ... 2. ...).\n"
    "- Max 2 sub-questions.\n"
    "- For 'X's Y's Z' chains: first find X or the bridge entity, then ask about Z.\n"
    "- For comparison (which film/person is earlier/later/born first): "
    "sub-Q1 finds attribute for option A, sub-Q2 finds attribute for option B.\n"
    "- Do NOT invent unrelated sub-questions (e.g. mother when asking about wife)."
)

MULTIHOP_DECOMPOSE_USER_EN = "Question: {query}\n\nSub-questions:"

MULTIHOP_SOLVE_SYSTEM_EN = (
    "You solve multi-hop questions using ONLY the provided documents. "
    "Documents may contain noise — ignore facts that do not match across reliable docs.\n"
    "Follow the sub-questions in order. Then output the final answer.\n"
    "Rules for FINAL answer:\n"
    "- Output a short phrase (name, place, date, yes/no only if question is yes/no).\n"
    "- For 'which film/person' questions: output the NAME of the chosen option, NOT 'yes'.\n"
    "- For birthplace/location: output the place name.\n"
    "Format:\n"
    "Hop1: <short fact>\n"
    "Hop2: <short fact if any>\n"
    "<answer>...</answer>"
)

MULTIHOP_SOLVE_USER_EN = (
    "Documents:\n{context}\n\n"
    "Original question: {query}\n\n"
    "Sub-questions (in order):\n{subquestions}\n\n"
    "Solve hops then final answer:"
)

# legacy hop prompts removed — kept for reference in git history
MULTIHOP_HOP_SYSTEM_EN = MULTIHOP_SOLVE_SYSTEM_EN
MULTIHOP_HOP_USER_EN = MULTIHOP_SOLVE_USER_EN
MULTIHOP_FINAL_SYSTEM_EN = MULTIHOP_SOLVE_SYSTEM_EN
MULTIHOP_FINAL_USER_EN = MULTIHOP_SOLVE_USER_EN


def format_context(docs: list[str], *, max_chars_per_doc: int = 1500, language: str = "zh") -> str:
    """把若干文档拼成带编号的上下文块，并裁剪过长文档。"""
    label = "Doc" if language == "en" else "文档"
    parts: list[str] = []
    for i, d in enumerate(docs):
        text = d.strip()
        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc] + "..."
        parts.append(f"[{label}{i}] {text}")
    return "\n\n".join(parts)


def naive_user_tmpl(*, language: str = "zh") -> str:
    return NAIVE_USER_TMPL if language == "zh" else NAIVE_USER_TMPL_EN


def build_naive_prompt(
    query: str,
    docs: list[str],
    *,
    language: str = "zh",
    dataset: str | None = None,
) -> tuple[str, str]:
    profile = get_naive_profile(dataset)
    sys_msg = profile.system(language=language)
    user_msg = profile.user_tmpl(language=language).format(
        query=query, n=len(docs), context=format_context(docs, language=language)
    )
    return sys_msg, user_msg


def build_closed_book_prompt(query: str, *, language: str = "zh") -> tuple[str, str]:
    sys_msg = CLOSED_BOOK_SYSTEM_ZH if language == "zh" else CLOSED_BOOK_SYSTEM_EN
    user_tmpl = CLOSED_BOOK_USER_TMPL_ZH if language == "zh" else CLOSED_BOOK_USER_TMPL_EN
    return sys_msg, user_tmpl.format(query=query)


def build_judge_prompt(
    query: str,
    pred: str,
    golds: list[str],
    *,
    language: str = "zh",
    dataset: str | None = None,
) -> tuple[str, str]:
    profile = get_judge_profile(dataset)
    gold_str = " / ".join(g for g in golds if g) if golds else ""
    user_tmpl = profile.user_tmpl(language=language)
    return profile.system(language=language), user_tmpl.format(
        query=query, gold=gold_str, pred=pred
    )


# 兼容旧引用
JUDGE_USER_TMPL = JUDGE_USER_TMPL_ZH
