# NLP 实验四 · 题目8

> **面向大模型 RAG 推理场景噪音文档的鲁棒性推理方法**

## 1. 项目目标

研究当 RAG 系统检索到**语义相关但缺乏逻辑依赖**的文档时，LLM 的推理行为会如何变化，并提出可检测、可矫正的鲁棒性推理方法。

详细计划见 [`PLAN.html`](./PLAN.html)。

## 2. 目录结构

```text
D:\code\nlpexp4\
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

# 5. 导出课程要求的标准输入/输出/参考答案文件
python scripts/export_standard_io.py

# 6. 启动 Demo
python demo/app.py
```

## 4. 数据与交付格式

本项目使用 RGB 数据集作为公开问答/RAG 数据源。每条样本包含 `query`、`answer`、`positive`、`negative`，其中 `zh_fact/en_fact` 额外包含 `positive_wrong` 和 `fakeanswer`。实验中不重新爬取网页或 PDF，而是把 RGB 已标注好的候选文档池建模为“检索结果”，通过受控采样研究噪音文档对 RAG 推理的影响。

噪音定义如下：

- `positive`：可支持答案的有效文档；
- `negative`：语义相关但不能支持答案的噪音文档；
- `positive_wrong`：表面支持错误答案的反事实噪音文档。

为对齐实验要求，`scripts/export_standard_io.py` 会把最新实验结果导出为：

- `input.json`：`id / question / context`，并额外保留 `documents / document_labels / noise_*` 方便复查；
- `output.json`：默认选取 token-F1 最高的方法作为主系统输出；
- `reference.json`：与输入 id 对齐的参考答案；
- `output_<method>.json` 与 `output_all_methods.json`：保留所有方法的横向对比输出。

当前已导出的标准文件位于：

```text
data/processed/exp4_existing_methods_zh_main_semantic_20260524_200425/
```

## 5. 代码规范

- **配置集中**：所有参数在 `src/config.py` 的 `Config` dataclass
- **接口统一**：所有矫正方法继承 `correctors.base.BaseCorrector`
- **可复现**：`config.seed = 42`，结果自动带时间戳保存 JSON
- **类型标注**：所有公开函数加 type hint
- **文件精简**：单文件 ≤ 200 行，超出即拆分
- **实验脚本只组装不实现**：核心逻辑全部下沉到 `src/` 模块

## 6. 5 个自主鲁棒性指标

| 指标 | 含义 |
|---|---|
| **NS**（Noise Sensitivity）| 噪音引起的相对性能下降幅度 |
| **NRS**（Noise Resistance Slope）| 噪音比例-性能曲线斜率 |
| **ISR**（Info-Source Rate）| 答案信息可溯源至 positive 文档的比例 |
| **NAR**（Noise Adoption Rate）| 答案中来自 negative 文档的信息占比 |
| **CRR**（Correction Recovery Rate）| 矫正机制对噪音损失的恢复程度 |

## 7. 关键 Deadline

| 时间 | 事项 |
|---|---|
| 5月26日 | 中期检查（设计方案 + 架构图） |
| 6月26日 | 系统演示（Gradio Demo） |
| 6月30日 | 最终提交（代码 + 报告 zip） |
