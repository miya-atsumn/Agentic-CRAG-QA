import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
#os.environ["HF_HUB_OFFLINE"] = "1" 
import json
import pandas as pd
import sys
import types
import importlib

# --- 兼容性垫片：修复 ragas 导入报错 ---
_VERTEXAI = "langchain_community.chat_models.vertexai"
if _VERTEXAI not in sys.modules:
    _stub = types.ModuleType(_VERTEXAI)
    _stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules[_VERTEXAI] = _stub
    setattr(importlib.import_module("langchain_community.chat_models"), "vertexai", _stub)

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ================= 1. 环境与模型初始化 =================
load_dotenv()

# 评判员 LLM（用 DeepSeek 来打分）
judge_llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0
)

# 加载本地向量库
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

PDF_PATH = "test.pdf"
DB_DIR = "./chroma_db_eval" # 换个新名字，避免和之前的空库冲突

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
    print("⏭️ 检测到已有的向量库，直接加载...")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
else:
    print("🛠️ 没有找到向量库，正在读取 PDF 并重新构建...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ PDF 解析完成，共切分 {len(chunks)} 个片段。正在写入数据库...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print("✅ 向量库构建完成！")

# 看看库里到底有多少东西
print(f"📊 当前向量库中包含 {vectorstore._collection.count()} 个文档片段")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 基础生成 Prompt
gen_prompt = ChatPromptTemplate.from_template(
    "你是一个严谨的助手。请只依据以下资料回答问题。如果资料没有，请说不知道。\n\n资料：{context}\n\n问题：{question}"
)
gen_chain = gen_prompt | judge_llm | StrOutputParser()

# ================= 2. 定义 Baseline RAG =================
def run_baseline_rag(question):
    """基线系统：纯向量检索直接生成"""
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])
    answer = gen_chain.invoke({"context": context, "question": question})
    return answer, [d.page_content for d in docs]

# ================= 3. 定义 CRAG (简化版) =================

def run_crag_rag(question):
    """CRAG系统：检索 -> 评估 -> 如果不好就重写查询 -> 重新检索 -> 生成"""
    # 1. 初始检索
    docs = retriever.invoke(question)
    
    # 防御性判断
    if not docs:
        print(f"  ⚠️ [CRAG] 初始检索为空: {question}")
        return "未检索到相关文档。", []
        
    # 2. 评估器（Grader）：让 LLM 判断检索到的文档有没有用
    grader_prompt = ChatPromptTemplate.from_template(
        "问题：{question}\n检索到的文档：{doc}\n\n请判断这段文档是否能回答该问题？只输出 yes 或 no。"
    )
    grader_chain = grader_prompt | judge_llm | StrOutputParser()
    
    is_relevant = grader_chain.invoke({"question": question, "doc": docs[0].page_content}).strip().lower()

    # 3. 查询重写（Query Rewriting）
    if "no" in is_relevant:
        # 如果评估为不相关，让 LLM 重写查询词
        rewriter_prompt = ChatPromptTemplate.from_template(
            "用户问了这个问题：{question}\n但之前的检索失败了。请提取问题中的核心实体（如人名、书名、特定名词），生成一个更利于搜索的短查询词。只输出查询词，不要解释。"
        )
        rewriter_chain = rewriter_prompt | judge_llm | StrOutputParser()
        new_query = rewriter_chain.invoke({"question": question})
        print(f"  🔄 [CRAG] 触发重写！原问题: {question} -> 新查询: {new_query}")

        # 使用新查询重新检索
        docs = retriever.invoke(new_query)
        
        # 重写后再次判断，如果还是空，直接返回
        if not docs:
            return "重写查询后仍未检索到相关文档。", []
            
    context = "\n\n".join([d.page_content for d in docs])
    answer = gen_chain.invoke({"context": context, "question": question})
    return answer, [d.page_content for d in docs]

# ================= 4. 执行评估 =================
def main():
    # 读取测试集
    with open("test_set.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 收集 Baseline 数据
    print("🚀 正在运行 Baseline RAG (基线系统)...")
    baseline_results = []
    for item in test_data:
        ans, ctxs = run_baseline_rag(item["question"])
        baseline_results.append({
            "user_input": item["question"],
            "response": ans,
            "retrieved_contexts": ctxs,
            "reference": item["ground_truth"]
        })
        
    # 收集 CRAG 数据
    print("\n🚀 正在运行 CRAG (带评估与重写)...")
    crag_results = []
    for item in test_data:
        ans, ctxs = run_crag_rag(item["question"])
        crag_results.append({
            "user_input": item["question"],
            "response": ans,
            "retrieved_contexts": ctxs,
            "reference": item["ground_truth"]
        })

    # 转换为 RAGAS 需要的 Dataset 格式
    baseline_dataset = Dataset.from_list(baseline_results)
    crag_dataset = Dataset.from_list(crag_results)

    # 定义四大指标
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    print("\n⏳ 正在让 RAGAS 评估 Baseline... (这步会调用多次 API，请耐心等待)")
    baseline_eval = evaluate(dataset=baseline_dataset, metrics=metrics, llm=judge_llm, embeddings=embeddings)
    
    print("\n⏳ 正在让 RAGAS 评估 CRAG... (这步会调用多次 API，请耐心等待)")
    crag_eval = evaluate(dataset=crag_dataset, metrics=metrics, llm=judge_llm, embeddings=embeddings)

    # 生成对比表格
    metrics_list = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    
    baseline_df = baseline_eval.to_pandas()[metrics_list].mean()
    crag_df = crag_eval.to_pandas()[metrics_list].mean()
    
    comparison = pd.DataFrame({
        "Baseline RAG": baseline_df,
        "CRAG (优化后)": crag_df,
        "提升幅度": crag_df - baseline_df
    }).round(3)

    print("\n" + "="*50)
    print("📊 RAG 系统量化评估报告")
    print("="*50)
    print(comparison)
    print("="*50)
    
    # 保存结果
    comparison.to_csv("evaluation_report.csv", encoding="utf-8-sig")
    print("✅ 报告已保存至 evaluation_report.csv")

if __name__ == "__main__":
    main()