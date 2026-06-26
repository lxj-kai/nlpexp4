# NLP 实验四 · 题目 8

> **面向大模型 RAG 推理场景噪音文档的鲁棒性推理方法**

研究当 RAG 系统检索到**语义相关但缺乏逻辑依赖**的文档时，LLM 的推理行为如何变化，并提出可检测、可矫正的鲁棒性推理方法。

## 1. 目录结构

本目录即完整项目根，可直接在此配置环境并运行：

```text
├── README.md
├── src/                 # 核心模块（数据加载、噪音注入、RAG、评测、矫正方法）
├── experiments/         # 实验脚本与 results/ 运行结果
├── backend/             # FastAPI 后端
├── frontend/            # Vue 3 交互演示
├── data/                # RGB / 2Wiki / Cmedqa / MobileMem 等数据集
├── figures/             # 实验图表
├── report_final/        # 终期报告（main.pdf + LaTeX 源文件）
├── scripts/             # 出图与结果查看脚本
├── tests/
├── requirements.txt
└── .env.example
```

## 2. 环境配置

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# LMSTUDIO_*  — 本地 LM Studio（OpenAI 兼容接口）
# DEEPSEEK_*  — DeepSeek API（LLM Judge，可选）
```

验证安装：

```bash
python -m src.smoke_test
python -m pytest -q
```

## 3. 查看实验结果

结果位于 `experiments/results/`，索引见 `experiments/results/INDEX.json`。

```bash
python scripts/show_exp1.py
python scripts/show_exp2.py
python scripts/show_exp3.py
python scripts/show_exp4.py
```

**终期报告**：直接打开 `report_final/main.pdf`。

重新编译 LaTeX（需 XeLaTeX）：

```bash
cd report_final
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```

## 4. 运行实验

通用参数：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--n` | 样本数 | 50 |
| `--language` | zh / en | zh |
| `--subset` | main / fact / refine / int | main |
| `--noise-type` | semantic / counterfactual / mixed | semantic |

```bash
# 实验一：噪音影响
python -m experiments.exp1_noise_impact --n 50 --language zh --subset main

# 实验二：矫正方法对比
python -m experiments.exp2_correction --n 50 --language zh --subset main

# 实验三：案例分析
python -m experiments.exp3_case_study --n 50 --pick 20 --language zh --subset main

# 实验四：现有方法横向对比
python -m experiments.exp4_existing_methods --n 50 --language zh --subset main \
  --noise-type semantic --ratio 0.75

# 实验五：深度实验
python -m experiments.exp5_deep --n 50 --language zh
```

生成图表：

```bash
python scripts/render_all_figures.py
python scripts/render_demo_figures.py
python scripts/render_late_stage_figures.py
```

> 复现实验需配置 LM Studio / DeepSeek API。

## 5. 交互演示（Vue + FastAPI）

**后端：**

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

**前端：**

```bash
cd frontend
npm install
npm run dev
```

浏览器打开 http://127.0.0.1:5173 。支持数据集：`rgb`、`2wiki`、`cmedqa`、`mobilemem`。

## 6. 方法概览

**五个鲁棒性指标：** NS · NRS · ISR · NAR · CRR（见 `src/metrics.py`）

**十二种矫正方法：** `naive` · `prompt` · `iterative` · `confidence` · `selfrag` · `voting` · `adaptive` · `iterative_sc` · 四个 `ablated_*` 消融变体（见 `src/correctors/`）

## 7. 常见问题

| 现象 | 处理 |
|------|------|
| 依赖缺失 | 激活 venv 后 `pip install -r requirements.txt` |
| LM Studio 连接失败 | 检查 `.env` 中 `LMSTUDIO_API_BASE`，确认模型已加载 |
| 前端无法访问 API | 先启动后端 8000，再启动 Vite 5173 |
| 前端依赖缺失 | `cd frontend && npm install` |
