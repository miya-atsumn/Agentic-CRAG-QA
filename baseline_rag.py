import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ================= 1. 环境配置 =================
load_dotenv()  # 从 .env 文件读取 DEEPSEEK_API_KEY

# ================= 2. 加载 PDF 并切分 =================

PDF_PATH = "test.pdf" 

def load_and_split_pdf(file_path: str):
    """加载 PDF，并按中文习惯进行递归切分"""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # 参考 kb_qa 的中文切分参数：chunk_size=500, overlap=100
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ PDF 加载完成，共切分为 {len(chunks)} 个片段")
    return chunks

# ================= 3. 向量化与构建向量库 =================
def build_vectorstore(chunks):
    """使用本地 BGE 模型进行向量化，并存入 Chroma"""
    # 使用中文优化的 bge-small-zh-v1.5 模型
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},  # 如有 GPU 可改为 'cuda'
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 构建 Chroma 向量库，并持久化到本地目录
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("✅ 向量库构建完成，已持久化到 ./chroma_db")
    return vectorstore

# ================= 4. 构建 RAG 链 =================
def build_rag_chain(vectorstore):
    """组装检索器、提示词和 LLM，形成完整的 RAG 链"""
    # 检索器：每次返回最相关的 3 个片段
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 初始化 DeepSeek LLM（兼容 OpenAI SDK）
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0  # 低温度减少幻觉
    )
    
    # 提示词模板：强制模型“只依据参考资料回答”
    prompt = ChatPromptTemplate.from_template(
        """你是一个严谨的文档问答助手。请**只依据**以下参考资料回答问题。
如果参考资料中没有相关信息，请如实回答“根据现有资料，我无法回答这个问题”。

参考资料：
{context}

问题：{question}
"""
    )
    
    # 格式化文档：将检索到的片段拼接成字符串
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # 构建 LCEL 链：检索 → 格式化 → 填充提示词 → LLM → 解析输出
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# ================= 5. 主程序 =================
def main():
    # 步骤 A：加载并切分 PDF
    chunks = load_and_split_pdf(PDF_PATH)
    
    # 步骤 B：构建向量库（若已存在可跳过，直接加载）
    if os.path.exists("./chroma_db"):
        print("⏭️  检测到已有向量库，直接加载...")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
    else:
        vectorstore = build_vectorstore(chunks)
    
    # 步骤 C：构建 RAG 链
    rag_chain = build_rag_chain(vectorstore)
    
        # 步骤 D：交互式问答
    print("\n🤖 RAG 问答系统已就绪（输入 'quit' 退出）\n")
    while True:
        question = input("❓ 你的问题：")
        if question.lower() == "quit":
            break
        
        print("🔍 正在检索并生成答案...")
        
        # --- 探照灯 ---
        # 看看检索器到底找出了什么
        docs = vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(question)
        print("\n【调试信息】检索到的原文片段：")
        # 防御空列表
        if not docs:
            print("  ⚠️ 没有任何检索结果！")
        else:
            for i, doc in enumerate(docs):
                print(f"片段 {i+1}: {doc.page_content[:100]}...")
        print("-" * 30)
        # -----------------------

        answer = rag_chain.invoke(question)
        print(f"\n💡 答案：{answer}\n")
        print("-" * 50)

if __name__ == "__main__":
    main()