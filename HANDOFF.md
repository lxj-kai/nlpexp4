# nlpexp4 交接记忆（给下一个 AI）

> 更新：2026-05-16 · 主任务：**Gradio → Vue 3 + FastAPI**

## 项目位置

- **仓库**：`D:\code\nlpexp4` · GitHub `https://github.com/lxj-kai/nlpexp4`
- **另一项目（未做完）**：`D:\mem2_out` — uid 0–9 缺 `type=friend` 的 stage10 记录，stage7_3 只跑了 10–19

## nlpexp4 是什么

NLP 实验四：RGB 数据集 + DeepSeek API，研究 **RAG 检索噪音**（比例 / 类型 / 位置）对答案的影响，对比 naive RAG 与多种 **corrector**。

### 曾发生的严重问题

`src/` 下 6 个核心 `.py` 一度只剩 `__pycache__/*.cpython-310.pyc`（decompyle3 不支持 3.10）。已从字节码反汇编**重建**：

`config.py`, `utils.py`, `data_loader.py`, `llm_client.py`, `noise_injector.py`, `rag_pipeline.py`

其余源码一直存在：`evaluator.py`, `metrics.py`, `prompts.py`, `correctors/`, `experiments/`, `demo/app.py`。

### 关键概念

- **`noise_ratio`**：注入文档中噪音占比（semantic / counterfactual / mixed），不是模型原始输入比例 alone。
- **`NoisyContext`**：`docs` + `labels`（positive / negative / positive_wrong）。
- **评测**：EM, Contains, Token-F1, ROUGE-L, ISR, NAR（`Evaluator(use_llm_judge=False)` 在 demo 里）。

### 实验状态（2026-05-16）

已跑：exp1（zh main/fact/position）、exp2（correction）、exp4（baselines）；样本常 10–20 条。  
**未跑**：exp3 case study；全量 300 条；英文实验。

### Gradio Demo（参考实现，保留）

- 路径：`demo/app.py` · 端口 **7861**（7860 曾被僵尸进程占用）
- 四阶段 UI：A 原始检索池 → B 注入后文档 → C Prompt → D 答案+指标
- 字体：Source Serif / Noto Serif SC；`html.escape` 渲染文档卡片

### Vue 迁移（进行中）

| 路径 | 状态 |
|------|------|
| `backend/main.py` | FastAPI：`/api/config`, `/samples`, `/sample/{id}`, `/inject`, `/run` |
| `frontend/` | Vue 3 + Vite + axios，四 Stage 单页，学术风 CSS |
| `backend/__init__.py` | 占位 |

**启动（待验证）：**

```bash
# 后端（项目根目录）
cd D:\code\nlpexp4
uvicorn backend.main:app --reload --port 8000

# 前端
cd D:\code\nlpexp4\frontend
npm install
npm run dev   # http://127.0.0.1:5173 ，/api 代理到 8000
```

**环境**：Windows · conda Python 3.10 · `.env` 含 `DEEPSEEK_API_KEY`（勿提交）。  
Gradio 依赖约束：`huggingface_hub<1.0`, `starlette~=0.40`, `fastapi~=0.115`。

### 待办（Vue 任务）

1. `npm install` + 联调前后端
2. 修 App.vue / backend HTML 若仍有 `motion.div` 笔误
3. README 增加 Vue 启动说明
4. 用户要求时再 `git commit` + push

## mem2 朋友圈缺口（用户曾问，未执行恢复）

- `stage7_3_tickets.jsonl` 仅 uid 10–19 有 friend（198 行）
- uid 0–9：stage4 有 friend_info（152 events），`stage10_image_summaries.jsonl` 有 180 条 vision 摘要，磁盘有 PNG
- 恢复思路：用 stage4 对齐 PNG → 补 stage7_3 / stage10；或 vision 摘要反推

## Vexty MCP（本 workspace `mcp-4`）

- 用户可见回复须走 **`check_messages` 的 `summary`**，勿在正文直接长回复
- 每轮回复后再 `check_messages` 等待下一条

## 用户偏好

- 前端效果 Gradio 已可接受，要求 **用 Vue 重写**
- 短答场景只给答案，不扩写
- 不要主动 commit，除非用户要求
