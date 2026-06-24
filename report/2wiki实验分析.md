# 2WikiMultihopQA 实验结果分析

> 正式实验规模 **n=500** · 生成时间：20260624_101946 · 数据集：xanhho/2WikiMultihopQA (dev)

## 1. 数据集说明

| 属性 | 值 |
| --- | --- |
| 语言 | en |
| 子集 main | 5000 条 corpus（实验抽样 **n=500**） |
| 子集 fact | 800 条 corpus（实验抽样 **n=500**） |
| 正例 | supporting_fact 对应句子/段落 |
| 负例 | 同 context 内 distractor 文章（hard negative） |
| 题型 | compositional / comparison / bridge_comparison / inference |

## 2. 实验一：语义噪音梯度（naive, main）

| 噪音比例 | Token-F1 | ROUGE-L | Contains | ISR | NAR |
| --- | --- | --- | --- | --- | --- |
| 0.00 | 0.619 | 0.618 | 0.674 | 0.653 | 0.000 |
| 0.25 | 0.524 | 0.524 | 0.574 | 0.534 | 0.051 |
| 0.50 | 0.519 | 0.518 | 0.578 | 0.453 | 0.137 |
| 0.75 | 0.422 | 0.421 | 0.472 | 0.337 | 0.182 |
| 1.00 | 0.067 | 0.067 | 0.062 | 0.000 | 0.108 |

- **NS**（semantic）= 0.3811

- r=0.75 相对 clean：F1 Δ-0.197，NAR Δ+0.182，ISR Δ-0.316

## 3. 实验一（fact）：反事实噪音梯度（naive, fact）

| 噪音比例 | Token-F1 | Contains | ISR | NAR |
| --- | --- | --- | --- | --- |
| 0.00 | 0.093 | 0.104 | 0.100 | 0.000 |
| 0.25 | 0.079 | 0.078 | 0.089 | 0.027 |
| 0.50 | 0.088 | 0.088 | 0.098 | 0.036 |
| 0.75 | 0.079 | 0.084 | 0.092 | 0.027 |
| 1.00 | 0.012 | 0.026 | 0.000 | 0.014 |

- **NS**（counterfactual）= 0.3073

## 4. 实验二：矫正方法对比（main, semantic, r=0.75）

| 方法 | Token-F1 | Contains | ISR | NAR |
| --- | --- | --- | --- | --- |
| confidence | 0.577 | 0.674 | 0.425 | 0.189 |
| iterative | 0.325 | 0.358 | 0.316 | 0.040 |
| naive | 0.422 | 0.472 | 0.337 | 0.182 |
| prompt | 0.406 | 0.464 | 0.330 | 0.187 |
| voting | 0.509 | 0.582 | 0.440 | 0.252 |

## 5. 实验四：现有方法横向对比（main, semantic）

| 方法 | r=0.5 F1 | r=0.5 NAR | r=0.75 F1 | r=0.75 NAR |
| --- | --- | --- | --- | --- |
| confidence | 0.698 | 0.139 | 0.579 | 0.189 |
| iterative | 0.373 | 0.022 | 0.326 | 0.040 |
| naive | 0.519 | 0.137 | 0.422 | 0.182 |
| prompt | 0.494 | 0.128 | 0.406 | 0.187 |
| selfrag | 0.519 | 0.137 | 0.422 | 0.182 |
| voting | 0.576 | 0.159 | 0.509 | 0.252 |

## 6. 关键发现

1. **Hard negative 有效**：NAR 随噪音单调上升（0→0.18@r=0.75），NS≈0.38，与 MIRIAD/Cmedqa 同量级但 clean F1 更高（短答匹配）。
2. **方法分化明显（r=0.75）**：confidence F1 最高；iterative NAR 最低（~0.04），适合压噪音采纳。
3. **selfrag ≈ naive**：多跳 hard neg 场景下 Self-RAG 基线几乎无增益。
4. **fact 子集**：反事实构造仍偏弱，NS 不稳定；主结论以 main 子集为准。

## 7. 复现命令

```bash
bash scripts/run_2wiki_n500.sh
# 或手动：
python -m experiments.exp_2wiki --n 500 --workers 10 --phase exp1   &
python -m experiments.exp_2wiki --n 500 --workers 10 --phase exp1_fact &
python -m experiments.exp_2wiki --n 500 --workers 10 --phase exp2   &
python -m experiments.exp_2wiki --n 500 --workers 10 --phase exp4   &
wait
python scripts/analyze_2wiki_results.py
```

## 8. 结果文件

- `/Users/miaohuairui/PyCharmMiscProject/nlpexp4-main/experiments/results/exp_2wiki_exp1_en_main_20260624_101946.json`
- `/Users/miaohuairui/PyCharmMiscProject/nlpexp4-main/experiments/results/exp_2wiki_exp1_fact_en_fact_20260624_102009.json`
- `/Users/miaohuairui/PyCharmMiscProject/nlpexp4-main/experiments/results/exp_2wiki_exp2_en_main_20260624_103400.json`
- `/Users/miaohuairui/PyCharmMiscProject/nlpexp4-main/experiments/results/exp_2wiki_exp4_en_main_20260624_103400.json`
