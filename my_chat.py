from AutoMemoryAgent import create_memory_agent
import os
from langchain_tavily import TavilySearch
import os
from datetime import datetime
from qwen_config import llm


os.environ["TAVILY_API_KEY"] = "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"
# 创建原始智能体（与之前相同）
memory_plugin=["history","memory","tool","profile"]
#memory_plugin=["history","memory","profile"]

 
agent_id="agent_001"
user_id="user_123"
date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
system_prompt = f"你是一个多动症和ADHD的心理辅导师，和你对话的是一个患病儿童，请用友善，儿童化的语言和他对话。现在的时间是{date}"
main_function_tools = [TavilySearch(max_results=2)]

agent=create_memory_agent(user_id,agent_id,system_prompt,main_function_tools,memory_plugin)
pidan="我是一个多动症和ADHD的小孩，我今年7岁了，男孩，上小学一年级了，出生在北京，对新奇的事情感兴趣，喜欢体育和音乐"
response="你好，小朋友"
# 主对话循环
#while True:
#user_input=llm.invoke(pidan+f"别人和我说{response},我应该回答：直接返回回答的内容，不要有无关的信息").content
user_input="你好呀"
response = agent.invoke(user_input)
print ("------------------")
print ("皮蛋说",user_input)
print("智能体回复", response)
print ("\n\n")