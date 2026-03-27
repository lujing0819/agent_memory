'''
qwen_llm 的 Docstring
调用百炼平台的qwen
'''
 
from langchain_openai import ChatOpenAI
import os
from langchain_community.embeddings import DashScopeEmbeddings
llm = ChatOpenAI(
    model="qwen3.5-plus",
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv("api_key"),
)
safe_llm = ChatOpenAI(
    model="qwen3.5-35b-a3b",
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv("api_key")
)
mini_llm=safe_llm
os.environ["DASHSCOPE_API_KEY"]=os.getenv("api_key")
embedding_model = DashScopeEmbeddings(model="text-embedding-v3")