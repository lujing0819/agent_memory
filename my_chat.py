from AutoMemoryAgent import create_memory_agent
import os
from langchain_tavily import TavilySearch
import os




os.environ["TAVILY_API_KEY"] = "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"
# 创建原始智能体（与之前相同）
#memory_plugin=["history","memory","tool","profile"]
memory_plugin=["history","memory","profile"]

 
agent_id="agent_001"
user_id="user_123"
system_prompt = "你是一个多动症和ADHD的心里辅导师，和你对话的是一个患病儿童，请用友善，儿童化的语言和他对话，"
main_function_tools = [TavilySearch(max_results=2)]
agent=create_memory_agent(user_id,agent_id,system_prompt,main_function_tools,memory_plugin)

# 主对话循环
while True:
    user_input = input("请输入表达：")
    if user_input.lower() == "exit":
        break
    response = agent.invoke(user_input)
    print("智能体回复", response)