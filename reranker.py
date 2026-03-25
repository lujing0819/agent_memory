import dashscope
from http import HTTPStatus
import os
#os.environ["DASHSCOPE_API_KEY"]=os.getenv("api_key")
def text_rerank(query,documents,threshold=0.75):
    dashscope.api_key = os.getenv("api_key")
    resp = dashscope.TextReRank.call(
        model="qwen3-rerank",
        query=query,
        documents=documents,
        top_n=10,
        return_documents=True,
        instruct="Given a web search query, retrieve relevant passages that answer the query."

    )
    indexs=[ s.index for s in resp['output']["results"] if s.relevance_score>threshold]
    results=[ documents[i] for i in indexs]
    return results
if __name__ == '__main__':
    query="什么是文本排序模型"
    documents=[
        "文本排序模型广泛用于搜索引擎和推荐系统中，它们根据文本相关性对候选文本进行排序",
        "量子计算是计算科学的一个前沿领域",
        "预训练语言模型的发展给文本排序模型带来了新的进展"
    ]
    result=text_rerank(query,documents)
    print (result)