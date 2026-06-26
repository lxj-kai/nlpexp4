---
license: mit
task_categories:
- question-answering
- text-generation
language:
- en
size_categories:
- 1K<n<10K
tags:
- rag
- noise
- benchmark
- retrieval-augmented-generation
- llm-evaluation
---

# Dataset Card for NoiserBench

This dataset card describes NoiserBench, a comprehensive evaluation framework for analyzing the role of noise in Retrieval-Augmented Generation (RAG) systems with Large Language Models.

## Dataset Details

### Dataset Description

NoiserBench is a comprehensive benchmark designed to evaluate how different types of noise affect Large Language Models in Retrieval-Augmented Generation scenarios. The benchmark encompasses multiple datasets and reasoning tasks, specifically designed to analyze seven distinct noise types from a linguistic perspective. This framework reveals that noise can be categorized into two practical groups: beneficial noise (which may enhance model capabilities) and harmful noise (which generally impairs performance).

- **Language(s) (NLP):** English
- **License:** MIT
- **Paper:** [Pandora's Box or Aladdin's Lamp: A Comprehensive Analysis Revealing the Role of RAG Noise in Large Language Models](https://arxiv.org/abs/2408.13533)

### Dataset Sources

- **Repository:** https://github.com/jinyangwu/NoiserBench
- **Paper:** https://arxiv.org/abs/2408.13533

## Uses

NoiserBench is designed for:
- Evaluating the robustness of RAG systems under different noise conditions
- Analyzing how various noise types affect LLM performance in retrieval scenarios
- Benchmarking different LLM architectures and scales on noisy retrieval tasks
- Research into developing more robust and adaptable RAG solutions
- Understanding the distinction between beneficial and harmful noise in RAG contexts

## Dataset Structure

The benchmark encompasses multiple datasets and reasoning tasks designed to evaluate seven distinct noise types from a linguistic perspective. The framework categorizes noise into:

1. **Beneficial Noise**: Types of noise that may enhance model capabilities and overall performance
2. **Harmful Noise**: Types of noise that generally impair LLM performance

The evaluation framework includes various reasoning tasks to comprehensively assess how different LLM architectures respond to these noise categories.

## Citation

**BibTeX:**

```bibtex
@article{wu2024pandora,
  title={Pandora's Box or Aladdin's Lamp: A Comprehensive Analysis Revealing the Role of RAG Noise in Large Language Models},
  author={Wu, Jinyang and Che, Feihu and Zhang, Chuyuan and Tao, Jianhua and Zhang, Shuai and Shao, Pengpeng},
  journal={arXiv preprint arXiv:2408.13533},
  year={2024}
}
```

**APA:**

Wu, J., Che, F., Zhang, C., Tao, J., Zhang, S., & Shao, P. (2024). Pandora's Box or Aladdin's Lamp: A Comprehensive Analysis Revealing the Role of RAG Noise in Large Language Models. arXiv preprint arXiv:2408.13533.

## Glossary

- **RAG (Retrieval-Augmented Generation)**: A method that combines information retrieval with text generation to reduce hallucinations in large language models
- **Beneficial Noise**: Types of noise that may enhance certain aspects of model capabilities and overall performance
- **Harmful Noise**: Types of noise that generally impair LLM performance in RAG scenarios
- **NoiserBench**: The comprehensive evaluation framework established in this work

## Dataset Card Contact

For questions about this dataset card or the underlying benchmark, please refer to the code repository or contact me at wu-jy23@mails.tsinghua.edu.cn.