# NLP 实验四 · 题目8

> **面向大模型 RAG 推理场景噪音文档的鲁棒性推理方法**

## 1. 项目目标

研究当 RAG 系统检索到**语义相关但缺乏逻辑依赖**的文档时，LLM 的推理行为会如何变化，并提出可检测、可矫正的鲁棒性推理方法。

详细计划见 [`PLAN.html`](./PLAN.html)。

## 2. 目录结构

```text
D:\code\nlpexp4\
├── data/rgb/                  # RGB 数据集 (zh.json / en.json 等)
├── src/                       # 核心模块
│   ├── config.py              # 全局配置 (dataclass)
│   ├── data_loader.py         # RGB 数据加载
│   ├── noise_injector.py      # 噪音注入器（类型/比例/位置）
│   ├── llm_client.py          # Deepseek API 客户端
│   ├── rag_pipeline.py        # Naive RAG pipeline
│   ├── evaluator.py           # EM / F1 / ROUGE-L / LLM-Judge
│   ├── metrics.py             # 5 个自主鲁棒性指标
│   └── correctors/            # 矫正方法
│       ├── base.py
│       ├── prompt_corrector.py
│       ├── iterative_corrector.py
│       ├── confidence_corrector.py
│       └── selfrag_baseline.py
├── experiments/               # 实验脚本
│   ├── exp1_noise_impact.py
│   ├── exp2_correction.py
│   ├── exp3_case_study.py
│   ├── exp4_existing_methods.py
│   └── results/               # 自动落盘的 JSON 结果
├── demo/app.py                # Gradio 交互 Demo
├── figures/                   # 自动生成的图表
├── report/                    # 实验报告
├── requirements.txt
├── .env.example
└── PLAN.html                  # 项目计划
```

## 3. 快速开始

```bash
# 1. 安装依赖
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. 配置 API Key
copy .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY

# 3. Smoke Test（50 条样本跑通 baseline）
python -m src.smoke_test

# 4. 运行实验
python -m experiments.exp1_noise_impact
python -m experiments.exp2_correction
python -m experiments.exp3_case_study
python -m experiments.exp4_existing_methods

# 5. 启动 Demo
python demo/app.py
```

## 4. 代码规范

- **配置集中**：所有参数在 `src/config.py` 的 `Config` dataclass
- **接口统一**：所有矫正方法继承 `correctors.base.BaseCorrector`
- **可复现**：`config.seed = 42`，结果自动带时间戳保存 JSON
- **类型标注**：所有公开函数加 type hint
- **文件精简**：单文件 ≤ 200 行，超出即拆分
- **实验脚本只组装不实现**：核心逻辑全部下沉到 `src/` 模块

## 5. 5 个自主鲁棒性指标

| 指标 | 含义 |
|---|---|
| **NS**（Noise Sensitivity）| 噪音引起的相对性能下降幅度 |
| **NRS**（Noise Resistance Slope）| 噪音比例-性能曲线斜率 |
| **ISR**（Info-Source Rate）| 答案信息可溯源至 positive 文档的比例 |
| **NAR**（Noise Adoption Rate）| 答案中来自 negative 文档的信息占比 |
| **CRR**（Correction Recovery Rate）| 矫正机制对噪音损失的恢复程度 |

## 6. 关键 Deadline

| 时间 | 事项 |
|---|---|
| 5月26日 | 中期检查（设计方案 + 架构图） |
| 6月26日 | 系统演示（Gradio Demo） |
| 6月30日 | 最终提交（代码 + 报告 zip） |
