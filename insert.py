import os
from tqdm import tqdm
# 适配LangChain 1.2.10，导入路径正确（无变更，修正可能的拼写错误）
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from qwen_config import llm
# 加载环境变量（需提前创建.env文件，写入OPENAI_API_KEY=你的密钥）
from pathlib import Path
os.environ["DASHSCOPE_API_KEY"]=os.getenv("api_key")

# -------------------------- 步骤1：加载并处理文档 --------------------------
def load_document(file_path):
    """根据文件类型加载文档（支持txt/pdf/docx）"""
    if file_path.endswith(".txt") or file_path.endswith(".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("仅支持txt/pdf/docx格式文档")
    # 加载文档并返回文档对象列表
    documents = loader.load()
    return documents

def split_documents(documents):
    """分割文档为小片段（避免Token超限）"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 每个片段最大Token数（约800汉字）
        chunk_overlap=200,  # 片段重叠部分（保证上下文连贯）
        separators=["\n##", "\n###", "\n", "。", "！", "？", "；", "，"]  # 按手册的标题/标点分割
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"文档分割完成，共生成 {len(split_docs)} 个片段")
    return split_docs

# -------------------------- 步骤2：构建向量库和检索器 --------------------------
def build_vector_db(split_docs, persist_directory="./chroma_db"):
    """将分割后的文档向量化并存储到Chroma向量库"""
    embeddings = DashScopeEmbeddings(model="text-embedding-v3")
    vector_db = Chroma(persist_directory=persist_directory,embedding_function=embeddings)
     
    for data in tqdm(split_docs):
        content=data.page_content
        metadata=data.metadata
        prompt=f"{content} 请基于提供的文本内容，生成摘要"
        abstract=llm.invoke(prompt).content 
        metadata['content']=content
        doc = Document(page_content=abstract,metadata=metadata) 
        vector_db.add_documents([doc])
    vector_db.persist()
    return vector_db



 

# -------------------------- 主函数：运行整个流程 --------------------------
if __name__ == "__main__":
    # 替换为你的手册文档路径（txt/pdf/docx均可）
    user_id="123"
    agent_id="agent_001"
    path=f"context\\{user_id}\\{agent_id}\\documents\\"
    doc_path=Path(f"{path}\\books")
    files = [f.name for f in doc_path.iterdir() if f.is_file() and f.name .endswith(".md")]
    #print (files)
    for f in tqdm(files):
        doc_path=f"{path}\\books\\{f}"
        print (doc_path)
        docs = load_document(doc_path)
        split_docs = split_documents(docs)
        
        # 2. 构建向量库和检索器
        vector_db = build_vector_db(split_docs,persist_directory=f"{path}\\vector_db\\")
 
    # while True:
    #     question = input("\n请输入你的问题（输入q退出）：")
    #     if question.lower() == "q":
    #         break
    #     result=vector_db.search(question, search_type="similarity", k=5)
    #     print (result)