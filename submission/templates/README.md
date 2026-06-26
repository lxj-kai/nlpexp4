# NLP 实验四 · 终期提交包

> **题目 8：面向大模型 RAG 推理场景噪音文档的鲁棒性推理方法**  
> 打包日期：{{BUILD_DATE}} · 包名：`{{PACKAGE_NAME}}`

本目录为**独立提交包**，解压后可直接配置环境、复现实验、查看结果与演示系统，**不依赖**开发仓库或其它路径。

---

## 1. 提交内容一览

| 类别 | 路径 | 说明 |
|------|------|------|
| 核心代码 | `src/` | 数据加载、噪音注入、RAG 流水线、评测指标、12 种矫正方法 |
| 实验脚本 | `experiments/` | exp1–exp5 及扩展实验入口 |
| 实验结果 | `experiments/results/` | 全部 JSON 结果（含 INDEX.json 索引） |
| 交互演示 | `backend/` + `frontend/` | FastAPI + Vue 3 四阶段演示 |
| 数据集 | `data/` | RGB / 2Wiki / Cmedqa / MobileMem 已转换 JSON |
| 图表 | `figures/` | 实验可视化输出 |
| 实验报告 | `report_final/` | LaTeX 源文件 + 已编译 PDF |
| 测试 | `tests/` | pytest 单元测试 |
| 工具脚本 | `scripts/` | 一键跑实验、出图、结果整理 |

完整文件清单与 SHA256 校验见 **`SUBMISSION_MANIFEST.json`**。

---

## 2. 目录结构

```text
{{PACKAGE_NAME}}/
├── README.md                    ← 本文件
├── SUBMISSION_MANIFEST.json     ← 文件清单与校验
├── requirements.txt
├── .env.example
├── src/                         ← 核心模块
├── experiments/
│   ├── exp1_noise_impact.py
│   ├── exp2_correction.py
│   ├── …
│   └── results/                 ← 运行结果（JSON）
│       ├── INDEX.json
│       ├── midterm/             ← RGB 主实验 exp1–exp5
│       ├── dataset_2wiki/
│       ├── dataset_new/
│       ├── exp_noise_gradient/
│       └── …
├── backend/                     ← FastAPI 后端
├── frontend/                    ← Vue 3 前端
├── data/
│   ├── rgb/                     ← RGB 中英文
│   ├── 2wiki/                   ← 2WikiMultihopQA（JSON）
│   ├── cmedqa/                  ← 中文医学检索
│   ├── mobilemem/               ← 中文移动记忆
│   └── processed/               ← 标准 input/output/reference
├── figures/                     ← 图表
├── report_final/
│   ├── main.pdf                 ← 终期报告（可直接阅读）
│   ├── main.tex
│   └── figures/
├── scripts/
└── tests/
```

---

## 3. 环境配置（首次使用）

### 3.1 Python 依赖

```bash
cd {{PACKAGE_NAME}}

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3.2 模型与 API

```bash
cp .env.example .env
# 编辑 .env：
#   LMSTUDIO_*  — 本地 LM Studio 问答模型（OpenAI 兼容接口）
#   DEEPSEEK_*  — DeepSeek API（LLM Judge 审查，可选）
```

### 3.3 冒烟测试

```bash
python -m src.smoke_test
python -m pytest -q
```

---

## 4. 查看实验结果（无需重新跑）

所有结果已保存在 `experiments/results/`，按实验类型分子目录。

```bash
# 查看结果索引
cat experiments/results/INDEX.json

# 各实验结果摘要脚本
python scripts/show_exp1.py
python scripts/show_exp2.py
python scripts/show_exp3.py
python scripts/show_exp4.py
```

### 4.1 结果目录说明

| 子目录 | 内容 |
|--------|------|
| `midterm/` | RGB 数据集 exp1–exp5 主实验（中/英、main/fact 等） |
| `dataset_2wiki/` | 2WikiMultihopQA 专项实验 |
| `dataset_new/` | Cmedqa / MIRIAD 扩展实验 |
| `exp_noise_gradient/` | 多数据集噪音梯度分析 |
| `exp_closed_book/` | 闭卷基线 |
| `sanity/`、`smoke/` | 新数据集 sanity / smoke 评测 |

详细说明见 `experiments/results/README.md`。

### 4.2 阅读报告

- **PDF（推荐）**：直接打开 `report_final/main.pdf`
- **重新编译**（需 XeLaTeX）：

```bash
cd report_final
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```

---

## 5. 复现实验（可选）

> 复现需配置 LM Studio / DeepSeek API，会产生 API 调用费用。

通用参数：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--n` | 样本数 | 50 |
| `--language` | zh / en | zh |
| `--subset` | main / fact / refine / int | main |
| `--noise-type` | semantic / counterfactual / mixed | semantic |

### 5.1 代表性命令

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

# 实验五：深度实验（四阶段）
python -m experiments.exp5_deep --n 50 --language zh
```

### 5.2 一键批量（交互确认）

```bash
python scripts/run_all_full.py        # 交互模式
python scripts/run_all_full.py -y     # 跳过确认
python scripts/run_all_full.py --dry-run
```

新结果默认写入 `experiments/results/`，带时间戳文件名。

### 5.3 生成图表

```bash
python scripts/render_all_figures.py
python scripts/render_demo_figures.py
python scripts/render_late_stage_figures.py   # 更新 report_final/figures/late_*.png
```

---

## 6. 交互演示系统（Vue + FastAPI）

### 6.1 启动后端

```bash
source .venv/bin/activate
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

健康检查：`curl http://127.0.0.1:8000/api/health`

### 6.2 启动前端

```bash
cd frontend
npm install
npm run dev
```

浏览器访问 **http://127.0.0.1:5173**（Vite 将 `/api` 代理到后端 8000 端口）。

### 6.3 支持的数据集

演示系统内置：`rgb`、`2wiki`、`cmedqa`、`mobilemem`（对应 `data/` 下 JSON）。

---

## 7. 核心方法说明

### 7.1 五个自主鲁棒性指标

| 指标 | 含义 |
|------|------|
| **NS** | 噪音引起的相对性能下降 |
| **NRS** | 噪音比例–性能曲线斜率 |
| **ISR** | 答案信息可溯源至 positive 文档的比例 |
| **NAR** | 答案中来自 negative 文档的信息占比 |
| **CRR** | 矫正机制对噪音损失的恢复程度 |

### 7.2 十二种矫正方法

`naive` · `prompt` · `iterative` · `confidence` · `selfrag` · `voting` · `adaptive` · `iterative_sc` · 以及 4 个 `ablated_*` 消融变体。  
均位于 `src/correctors/`，通过 `@register_corrector()` 注册。

---

## 8. 未包含的内容（体积原因）

以下内容**未**打入提交包，不影响阅读结果与复现主实验：

| 排除项 | 原因 | 如需获取 |
|--------|------|----------|
| `data/miriad/` 全量 | ~7 GB parquet | HuggingFace 下载后运行 `prepare_miriad.py` |
| `data/*/raw/` parquet | 原始下载缓存 | 各 `scripts/prepare_*.py` |
| `data/bright/`、`data/tempo/` 等 | 扩展探索数据集 | 开发仓库脚本 |
| `.env`、`.cache/` | 密钥与 LLM 缓存 | 本地自行配置 |
| `frontend/node_modules/` | 可通过 npm install 恢复 | `cd frontend && npm install` |

---

## 9. 常见问题

| 现象 | 处理 |
|------|------|
| `ModuleNotFoundError` | 确认已激活 venv 并 `pip install -r requirements.txt` |
| LM Studio 连接失败 | 检查 `.env` 中 `LMSTUDIO_API_BASE`，确认本地模型已加载 |
| 前端无法请求 API | 先启动后端 8000，再启动 Vite 5173 |
| 数据集加载失败 | 确认 `data/<name>/` 下 JSON 存在（见第 8 节） |
| pytest 失败 | 多数测试不依赖 API；若 Judge 相关失败，可设 `use_llm_judge=False` |

---

## 10. 提交检查清单

- [ ] 解压到任意目录，路径中**无中文空格问题**（建议纯英文路径）
- [ ] `report_final/main.pdf` 可正常打开
- [ ] `experiments/results/INDEX.json` 存在且 `total_json` > 100
- [ ] `python -m pytest -q` 通过（或仅 API 相关用例跳过）
- [ ] （可选）演示系统前后端联调成功

---

**课程**：NLP 实验四 · 清华大学  
**题目**：面向大模型 RAG 推理场景噪音文档的鲁棒性推理方法
