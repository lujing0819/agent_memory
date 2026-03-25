
from qwen_config import llm
from memory_forget import forget  
import os
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage,SystemMessage
 
from langchain_tavily import TavilySearch
import os
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool,tool
from context import ContextList

os.environ["TAVILY_API_KEY"] = "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"
 



class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


tools = [TavilySearch(max_results=2)]

user_id="user_123"
agent_id="agent_001"
ctx_List=ContextList(["history","memory","tool","profile","document"],agent_id,user_id)
for ctx in ctx_List.ctx_list:
    read_tool=tool(ctx.read)
    tools.append(read_tool)
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=""   
)

# ---------- 主对话循环 ----------
systemprompt="你是一个多动症和ADHD的心里辅导师，和你对话的是一个患病儿童，请用友善，儿童化的语言和他对话，"

messages = [SystemMessage(content=systemprompt)]           
prev_msg_count = 0      
initial_state={"messages": messages}   
while True:
    user_input = input("请输入表达：")
    if user_input.lower() == "exit":
        break
    # 构造状态并调用 agent（假设 agent 已定义）
    user_msg = HumanMessage(content=user_input)
    initial_state = {"messages": messages + [user_msg]}
    print ("user_input:",user_input)
    final_state = agent.invoke(initial_state)   
    messages=final_state["messages"]
    print ("智能体回复",messages[-1].content)
    new_messages = messages[prev_msg_count:]  
    #存储记忆
    ctx_List.write(new_messages)
    #遗忘记忆
    messages=forget(messages)
    #print (messages)
    prev_msg_count = len(messages)  