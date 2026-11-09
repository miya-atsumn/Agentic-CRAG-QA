# 📚 Agentic CRAG Document QA System

> 一个基于 **Corrective RAG (CRAG)** 架构的智能文档问答系统。通过引入评估器（Grader）与查询重写（Query Rewriting）机制，解决传统 Naive RAG 在面临“元数据查询（如标题/摘要）”时的语义漂移与检索失效问题，显著降低大模型幻觉。

## ✨ 核心特性 (Features)

- **原生 ReAct 引擎**：不依赖复杂框架，手写实现 `Thought-Action-Observation` 循环，具备工具调用与异常回注能力。
- **CRAG 纠错架构**：引入文档相关性评估器，动态检测检索质量。当质量不达标时，自动触发 Query Rewriting（查询重写）并重新检索。
- **RAGAS 量化评估**：内置基于 RAGAS 的自动化评估流水线，对比基线与优化后的系统在四大核心指标上的表现。
- **本地 Embedding**：集成 `BAAI/bge-small-zh-v1.5` 模型，支持完全离线的文本向量化，保护数据隐私。

## 🏗️ 系统架构 (Architecture)

```text
[用户提问] 
    │
    ▼
[向量检索 (BGE + Chroma)] ──► 召回 Top-K 文档
    │
    ▼
[Grader (LLM 评估器)] ──► 判断文档是否相关？
    │
    ├── (Yes / 相关) ──────────────────┐
    │                                  │
    └── (No / 不相关) ──► [Query Rewriting] ──► [重新检索] ──┘
                                       │
                                       ▼
                             [DeepSeek 生成最终答案]
```
## 📊 量化评估结果 (Evaluation Results)

在自建的中英跨语言测试集（基于通信+AI Agent 英文顶会论文）上，使用 **RAGAS** 框架（以 DeepSeek 作为 Judge 模型）进行对比评估：

| 评估指标 (Metrics) | Baseline RAG (基线) | CRAG (本系统) | 提升幅度 |
| :--- | :---: | :---: | :---: |
| **Context Recall** (召回率) | 0.600 | 0.600 | - |
| **Answer Relevancy** (答案相关性) | 0.780 | **0.968** | 🚀 **+24.1%** |
| **Faithfulness** (忠实度 / 降低幻觉) | 0.950 | **0.960** | 📈 **+1.1%** |
| **Context Precision** (精准率) | 0.500 | **0.600** | 📈 **+20.0%** |
## 🚀 快速开始 (Quick Start)
1. 安装依赖
bash
pip install -r requirements.txt
(或者手动安装：pip install langchain langchain-community langchain-huggingface langchain-chroma langchain-openai pypdf sentence-transformers ragas datasets pandas python-dotenv)
2. 配置环境变量
在项目根目录创建 .env 文件，填入你的 DeepSeek API Key：

text
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxx
3. 准备数据与测试集
将你的目标 PDF 文件命名为 test.pdf 并放入项目根目录（或修改代码中的 PDF_PATH 变量）。

准备 test_set.json 测试集（格式参考仓库内示例）。

4. 运行评估
bash
python evaluate.py
程序将自动解析 PDF、构建向量库，并分别运行 Baseline 和 CRAG 系统，最终输出对比评估报告（并保存为 evaluation_report.csv）。

## 📂 目录结构 (Project Structure)
text
├── baseline_rag.py        # 基础 Naive RAG 运行脚本
├── evaluate.py            # CRAG 核心逻辑与 RAGAS 评估脚本
├── test_set.json          # 评测数据集（Question + Ground Truth）
├── requirements.txt       # 依赖清单
├── .env                   # 环境变量（需自行创建，已加入 gitignore）
└── chroma_db_eval/        # 本地向量数据库（运行后自动生成）
Created with ❤️ by [miya_atsumn]