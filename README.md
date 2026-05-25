# NLP 实验四 · 题目8

> **面向大模型 RAG 推理场景噪音文档的鲁棒性推理方法**

## 1. 项目目标

研究当 RAG 系统检索到**语义相关但缺乏逻辑依赖**的文档时，LLM 的推理行为会如何变化，并提出可检测、可矫正的鲁棒性推理方法。

详细计划见 [`PLAN.html`](./PLAN.html)。

## 2. 目录结构

```text
nlpexp4/
├── data/rgb/                  # RGB 数据集 (zh.json / en.json 等)
├── data/processed/            # 面向验收的 input/output/reference 标准 JSON
├── src/                       # 核心模块
│   ├── config.py              # 全局配置 (dataclass)
│   ├── data_loader.py         # RGB 数据加载
│   ├── noise_injector.py      # 噪音注入器（类型/比例/位置）
│   ├── llm_client.py          # Deepseek API 客户端
│   ├── rag_pipeline.py        # Naive RAG pipeline
│   ├── evaluator.py           # EM / F1 / ROUGE-L / LLM-Judge
│   ├── metrics.py             # 5 个自主鲁棒性指标
│   ├── metrics_semantic.py    # 语义级文档溯源（深化方向2）
│   ├── prompts.py             # 中英文 Prompt 模板
│   ├── visualize.py           # 12 个可视化函数
│   └── correctors/            # 12 个矫正方法
│       ├── base.py
│       ├── prompt_corrector.py         # 方法A：提示词优化
│       ├── iterative_corrector.py      # 方法B：迭代过滤
│       ├── confidence_corrector.py     # 方法C：CoT证据链
│       ├── voting_corrector.py         # 方法D：多Persona投票
│       ├── selfrag_baseline.py         # SelfRAG对比
│       ├── adaptive_corrector.py       # 方向1：自适应路由
│       ├── ablated_confidence.py       # 方向3：消融变体(4个)
│       └── iterative_self_correct.py   # 方向4：迭代自纠正
├── experiments/               # 实验脚本
│   ├── _runner.py             # 通用 runner
│   ├── exp1_noise_impact.py   # 噪音影响
│   ├── exp2_correction.py     # 矫正对比
│   ├── exp3_case_study.py     # 案例分析
│   ├── exp4_existing_methods.py # 现有方法对比
│   ├── exp5_deep.py           # 深度实验集成
│   └── results/               # 自动落盘的 JSON 结果
├── demo/app.py                # Gradio 交互 Demo
├── backend/                   # FastAPI 后端
├── frontend/                  # Vue 3 前端
├── figures/                   # 实验图表
├── report_latex/              # LaTeX 实验报告
├── tests/                     # 测试套件
├── requirements.txt
├── .env.example
└── PLAN.html                  # 项目计划
```

## 3. 快速开始

```bash
# 1. 安装依赖
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY

# 3. Smoke Test
python -m src.smoke_test

# 4. 运行测试套件
python -m pytest -q
```

## 4. 实验运行

所有实验脚本均位于 `experiments/` 目录，通过 `python -m experiments.<name>` 运行，通用参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--n` | 样本数 | 50 |
| `--language` | 语言 (zh/en) | zh |
| `--subset` | 数据子集 (main/fact/refine/int) | main |
| `--noise-type` | 噪音类型 (semantic/counterfactual/mixed) | semantic |
| `--ratio` / `--ratios` | 噪音比例 | 见各实验 |
| `--methods` | 矫正方法列表(逗号分隔) | 见各实验 |

### 4.1 实验一：噪音影响分析

**配置：** naive 方法 × 5 噪音比例 (0, 0.25, 0.5, 0.75, 1.0) × 3 噪音类型 (semantic/counterfactual/mixed)

```bash
# ---- 主矩阵 ----
# 中文 main 子集 (15 conditions = 5 ratio × 3 type)
python -m experiments.exp1_noise_impact --n 50 --language zh --subset main

# 中文 fact 子集 (counterfactual 场景)
python -m experiments.exp1_noise_impact --n 50 --language zh --subset fact

# 英文 main 子集
python -m experiments.exp1_noise_impact --n 50 --language en --subset main

# ---- 位置子实验 ----
# 只跑位置对比 (4 positions)，跳过主矩阵，节省 token
python -m experiments.exp1_noise_impact --n 50 --language zh --subset main \
  --position-only --position-ratio 0.75 --position-noise-type semantic

python -m experiments.exp1_noise_impact --n 50 --language zh --subset fact \
  --position-only --position-ratio 0.75 --position-noise-type counterfactual
```

### 4.2 实验二：矫正方法对比

**配置：** 5 方法 (naive/prompt/iterative/confidence/selfrag) × 4 噪音比例 (0, 0.25, 0.5, 0.75)，默认 noise-type=semantic

```bash
# 中文 main · semantic (20 conditions)
python -m experiments.exp2_correction --n 50 --language zh --subset main --noise-type semantic

# 中文 fact · counterfactual
python -m experiments.exp2_correction --n 50 --language zh --subset fact --noise-type counterfactual

# 英文 main · semantic
python -m experiments.exp2_correction --n 50 --language en --subset main --noise-type semantic

# 自定义方法和比例
python -m experiments.exp2_correction --n 50 --methods naive,confidence,voting
python -m experiments.exp2_correction --n 50 --ratios 0.0,0.5,0.75
```

### 4.3 实验三：案例深度分析

**配置：** clean + noisy + corrected 三组对比，在多个噪音比例 (0.5, 0.75, 1.0) 下循环

```bash
# 中文 main (1 clean + 3 ratio × 2 = 7 conditions)
python -m experiments.exp3_case_study --n 50 --pick 20 --language zh --subset main

# 中文 fact
python -m experiments.exp3_case_study --n 50 --pick 20 --language zh --subset fact

# 英文 main
python -m experiments.exp3_case_study --n 50 --pick 20 --language en --subset main

# 自定义矫正方法和比例
python -m experiments.exp3_case_study --n 50 --pick 30 \
  --ratios 0.5,0.75,1.0 --corrector voting
```

### 4.4 实验四：现有方法横向对比

**配置：** 6 方法 (naive/selfrag/iterative/confidence/prompt/voting) × 单一噪音比例和类型

```bash
# 互锁效应矩阵：6方法 × 3噪音类型 × 2比例 (即 6 个独立命令)
# Semantic (r=0.5 和 r=0.75)
python -m experiments.exp4_existing_methods --n 50 --language zh --subset main \
  --noise-type semantic --ratio 0.5
python -m experiments.exp4_existing_methods --n 50 --language zh --subset main \
  --noise-type semantic --ratio 0.75

# Counterfactual (使用 fact 子集)
python -m experiments.exp4_existing_methods --n 50 --language zh --subset fact \
  --noise-type counterfactual --ratio 0.5
python -m experiments.exp4_existing_methods --n 50 --language zh --subset fact \
  --noise-type counterfactual --ratio 0.75

# Mixed
python -m experiments.exp4_existing_methods --n 50 --language zh --subset main \
  --noise-type mixed --ratio 0.5
python -m experiments.exp4_existing_methods --n 50 --language zh --subset main \
  --noise-type mixed --ratio 0.75

# 英文对照 (至少跑一组)
python -m experiments.exp4_existing_methods --n 50 --language en --subset main \
  --noise-type semantic --ratio 0.75
```

### 4.5 实验五：深度实验

**配置：** 四阶段连续运行 — 自适应路由 / 语义溯源 / 消融对比 / 迭代自纠正

```bash
# 中文 (N=50, ratio=0.75)
python -m experiments.exp5_deep --n 50 --language zh

# 英文
python -m experiments.exp5_deep --n 50 --language en
```

### 4.6 一键运行全部实验

```bash
# 交互式确认后顺序执行 7 个 step（含 token 预估）
python scripts/run_all_full.py

# 非交互模式
python scripts/run_all_full.py -y

# 只跑指定 step
python scripts/run_all_full.py --only exp1_zh_main,exp2_zh

# 跳过指定 step
python scripts/run_all_full.py --skip exp1_en_main,exp2_en

# 预览计划（不执行）
python scripts/run_all_full.py --dry-run
```

### 4.7 实验运行状态

以下为当前已完成的实验覆盖矩阵（n=50 级别，最新结果截至 2026-05-25）：

| 实验 | 语言 | 子集 | 噪音类型 | 比例 | 方法数 | 状态 |
|------|------|------|----------|------|--------|------|
| Exp1 | zh | main | semantic/cf/mixed | 0,0.25,0.5,0.75,1.0 | 1 (naive) | ✅ 完成 |
| Exp1 | zh | fact | semantic/cf/mixed | 0,0.25,0.5,0.75,1.0 | 1 (naive) | ✅ 完成 |
| Exp1 | en | main | semantic/cf/mixed | 0,0.25,0.5,0.75,1.0 | 1 (naive) | ✅ 完成 |
| Exp1-pos | zh | fact | semantic | 0.5 | 1 (naive) | ✅ 完成 |
| Exp1-pos | zh | fact | counterfactual | 0.75 | 1 (naive) | ✅ 完成 |
| Exp2 | zh | main | semantic | 0,0.25,0.5,0.75 | 5 | ✅ 完成 |
| Exp2 | en | main | semantic | 0,0.25,0.5,0.75 | 5 | ✅ 完成 |
| Exp2 | zh | fact | counterfactual | 0,0.25,0.5,0.75 | 5 | ✅ 完成 |
| Exp3 | zh | main | semantic | 0,0.5,0.75,1.0 | 2 (naive+confidence) | ✅ 完成 |
| Exp3 | en | main | semantic | 0,0.5,0.75,1.0 | 2 (naive+confidence) | ✅ 完成 |
| Exp3 | zh | fact | semantic | 0,0.5,0.75,1.0 | 2 (naive+confidence) | ✅ 完成 |
| Exp4 | zh | main | semantic | 0.5, 0.75 | 6 | ✅ 完成 |
| Exp4 | zh | main | mixed | 0.5, 0.75 | 6 | ✅ 完成 |
| Exp4 | zh | fact | counterfactual | 0.5, 0.75 | 6 | ✅ 完成 |
| Exp4 | en | main | semantic | 0.75 | 6 | ✅ 完成 |
| Exp5 | zh | main | semantic+cf | 0.75 | 4阶段 | ✅ 完成 |
| Exp5 | en | main | semantic+cf | 0.75 | 4阶段 | ✅ 完成 |

> 全部 17 项实验已完成 (100%)。

## 5. 图表生成

```bash
# 生成所有实验图表
python scripts/render_all_figures.py

# 生成演示用核心图表 + 关键数字
python scripts/render_demo_figures.py

# 各实验单独查看
python scripts/show_exp1.py
python scripts/show_exp2.py
python scripts/show_exp3.py
python scripts/show_exp4.py
```

## 6. 系统启动

```bash
# Gradio Demo
python demo/app.py                    # http://127.0.0.1:7861

# Vue 前端 + FastAPI 后端
uvicorn backend.main:app --reload --port 8000
cd frontend && npm install && npm run dev   # http://127.0.0.1:5173
```

## 7. LaTeX 报告编译

```bash
cd report_latex
xelatex main.tex     # 第一次编译
xelatex main.tex     # 第二次编译（生成目录）
```

## 8. 代码规范

- **配置集中**：所有参数在 `src/config.py` 的 `Config` dataclass
- **接口统一**：所有矫正方法继承 `correctors.base.BaseCorrector`，通过 `@register_corrector()` 注册
- **可复现**：`config.seed = 42`，结果自动带时间戳保存 JSON
- **缓存机制**：sha1 磁盘缓存，重复 prompt 零成本
- **类型标注**：所有公开函数加 type hint
- **实验脚本只组装不实现**：核心逻辑全部下沉到 `src/` 模块

## 9. 5 个自主鲁棒性指标

| 指标 | 含义 |
|---|---|
| **NS**（Noise Sensitivity）| 噪音引起的相对性能下降幅度 |
| **NRS**（Noise Resistance Slope）| 噪音比例-性能曲线斜率 |
| **ISR**（Info-Source Rate）| 答案信息可溯源至 positive 文档的比例 |
| **NAR**（Noise Adoption Rate）| 答案中来自 negative 文档的信息占比 |
| **CRR**（Correction Recovery Rate）| 矫正机制对噪音损失的恢复程度 |

## 10. 12 个矫正方法

| 方法 | API | 核心思想 | 适用场景 |
|------|-----|---------|---------|
| naive | 1 | 直接拼接文档 | 基线 |
| prompt (A) | 1 | 要求识别噪音、标注来源 | 反事实最优 |
| iterative (B) | n+1 | 逐文档评分→过滤→生成 | 语义候选 |
| confidence (C) | 1 | 拆解需求→证据匹配→输出 | 语义最优 |
| selfrag | n+2 | 相关判定→生成→支撑校验 | 现有方法对比 |
| voting (D) | 3-4 | 3 Persona 投票聚合 | 最稳健 |
| adaptive | 2-4 | 噪音检测→路由最优方法 | 自适应 |
| iterative_sc | 3-7 | 生成→自检→修订循环 | 迭代优化 |
| ablated_full | 1 | C完整版（消融对照） | 因果分析 |
| ablated_no_decompose | 1 | 去掉信息拆解 | 因果分析 |
| ablated_no_evidence | 1 | 去掉证据匹配 | 因果分析 |
| ablated_no_tag | 1 | 去掉结构约束 | 因果分析 |

## 11. 关键 Deadline

| 时间 | 事项 |
|---|---|
| 5月26日 | 中期检查（设计方案 + 架构图） |
| 6月26日 | 系统演示（Gradio Demo） |
| 6月30日 | 最终提交（代码 + 报告 zip） |
