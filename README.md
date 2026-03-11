# IISc LLM Engineering Course

[![Course Website](https://img.shields.io/badge/Course-llm--engg.github.io-blue)](https://llm-engg.github.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A repository for the **Large Language Models – A Hands-on Approach** course by Yoginder Negi and Bhuthesh R (IISc).

Course website: [https://llm-engg.github.io/](https://llm-engg.github.io/)

---

## Course Description

LLMs have become mainstay of NLP and are transforming every domain, from software development, research, and business intelligence to education. However, deploying them efficiently remains a specialized engineering challenge.

This course provides an engineering-focused exploration of Large Language Models (LLMs). Participants will go from understanding transformer architectures and GPU internals to mastering fine-tuning, inference optimization, and large-scale deployment across GPUs, clusters, and edge devices. Through a theory-to-practice approach, including case studies, hands-on labs, and projects, learners will cover key topics such as model architecture, fine-tuning techniques, inference optimization, serving strategies, and applications in retrieval-augmented generation (RAG) and agentic systems.

## Learning Outcomes

By the end of this course, participants will be able to:

- **Understand LLM Architecture**: Master transformer architectures, attention mechanisms, and modern LLM variants (GPT-OSS, Qwen, Gemma, etc.).
- **Optimize Inference**: Implement efficient inference strategies including quantization, KV caching, and serving via inference engines like vLLM.
- **Fine-tune Models**: Apply various fine-tuning techniques including instruction tuning, PEFT techniques like LoRA, QLoRA etc and preference alignment.
- **Build RAG Systems**: Design and implement Retrieval-Augmented Generation pipelines.
- **Develop AI Agents**: Create tool-using agents with the ReAct framework.
- **Deploy at Scale**: Set up production-ready LLM serving infrastructure with cost optimization.
- **Multimodal Models**: Work with vision-language models and speech.
- **Evaluation**: Understand evaluation strategies for LLMs and RAG systems.

---

## Course Curriculum

| Week | Module | Topic Code | Topic |
|------|--------|------------|-------|
| 1 | LLM Foundations I | 1.1 | Orientation, Transformer Architecture |
| 1 | LLM Foundations I | 1.2 | GPT-2 |
| 2 | LLM Foundations II | 2.1 | Modern Architectures |
| 2 | LLM Foundations II | 2.2 | Mixture of Experts |
| 3 | GPU Basics | 3.1 | GPU Architecture Deep Dive |
| 3 | GPU Basics | 3.2 | Parallelism: Multi GPU, Multi Node |
| 4 | Inference | 4.1 | Inference Strategies |
| 4 | Inference | 4.2 | Inference Math and Bottlenecks |
| 5 | Efficient Inference & Quantization | 5.1 | Efficient Attention & KV Caching |
| 5 | Efficient Inference & Quantization | 5.2 | Quantization Fundamentals |
| 6 | Inference | 6.1 | Inference Engines and Multi GPU |
| 7 | RAG Fundamentals | 7.1 | RAG Fundamentals – Context Engineering, Embeddings, Search and Rerankers |
| 8 | RAG | 8.1 | Evaluating RAG |
| 8 | RAG | 8.2 | ReAct Framework: Thought → Action → Observation |
| 9 | Reasoning & Agents | 9.1 | Tool Calling, Agents |
| 9 | Reasoning & Agents | 9.2 | Fine Tuning for Tool Calling |
| 10 | Instruction Tuning & Alignment | 10.1 | Instruction Tuning |
| 10 | Instruction Tuning & Alignment | 10.2 | Alignment (RLHF, DPO etc) |
| 11 | RL & Reasoning | 11.1 | More RL |
| 11 | RL & Reasoning | 11.2 | Reasoning & Chain-of-Thought |
| 12 | Evaluation | 12.1 | Evaluation I |
| 12 | Evaluation | 12.2 | Evaluation II |

---

## Repository Structure

```
IISc-LLM/
├── week-01-llm-foundations-1/        # LLM Foundations I
│   ├── 1.1-transformer-architecture.ipynb
│   └── 1.2-gpt2.ipynb
├── week-02-llm-foundations-2/        # LLM Foundations II
│   ├── 2.1-modern-architectures.ipynb
│   └── 2.2-mixture-of-experts.ipynb
├── week-03-gpu-basics/               # GPU Basics
│   ├── 3.1-gpu-architecture.ipynb
│   └── 3.2-parallelism.ipynb
├── week-04-inference/                # Inference
│   ├── 4.1-inference-strategies.ipynb
│   └── 4.2-inference-math.ipynb
├── week-05-efficient-inference/      # Efficient Inference & Quantization
│   ├── 5.1-kv-caching.ipynb
│   └── 5.2-quantization.ipynb
├── week-06-inference-engines/        # Inference Engines
│   └── 6.1-inference-engines-multi-gpu.ipynb
├── week-07-rag-fundamentals/         # RAG Fundamentals
│   └── 7.1-rag-fundamentals.ipynb
├── week-08-rag/                      # RAG
│   ├── 8.1-evaluating-rag.ipynb
│   └── 8.2-react-framework.ipynb
├── week-09-reasoning-agents/         # Reasoning & Agents
│   ├── 9.1-tool-calling-agents.ipynb
│   └── 9.2-fine-tuning-tool-calling.ipynb
├── week-10-instruction-tuning/       # Instruction Tuning & Alignment
│   ├── 10.1-instruction-tuning.ipynb
│   └── 10.2-alignment-rlhf-dpo.ipynb
├── week-11-rl-reasoning/             # RL & Reasoning
│   ├── 11.1-more-rl.ipynb
│   └── 11.2-chain-of-thought.ipynb
├── week-12-evaluation/               # Evaluation
│   ├── 12.1-evaluation-1.ipynb
│   └── 12.2-evaluation-2.ipynb
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- [CUDA-capable GPU](https://developer.nvidia.com/cuda-gpus) (recommended for most labs)
- [Hugging Face account](https://huggingface.co/) (for model access)

### Installation

```bash
# Clone the repository
git clone https://github.com/Sparten-Ashvinee/IISc-LLM.git
cd IISc-LLM

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter lab
```

---

## Resources

- 📖 [Course Website](https://llm-engg.github.io/)
- 📅 [Course Schedule](https://llm-engg.github.io/schedule/)
- 📝 [Assignments & Labs](https://llm-engg.github.io/assignments/)
- 🤗 [Hugging Face Hub](https://huggingface.co/)
- ⚡ [vLLM](https://github.com/vllm-project/vllm)
- 📚 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
