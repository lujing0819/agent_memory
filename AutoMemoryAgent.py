from typing import List, Callable
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from qwen_config import llm
from memory_forget import forget  
from langchain_core.messages import BaseMessage, HumanMessage,SystemMessage
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from context import ContextList
class AutoMemoryAgent:
    """
    自动管理记忆的智能体包装器。
    在每次调用时自动存储新增消息，并对历史消息进行遗忘修剪。
    """
    def __init__(
        self,
        agent: Runnable,
        ctx_list: "ContextList",  # 假设的上下文列表类型
        forget_func: Callable[[List[BaseMessage]], List[BaseMessage]],
        system_prompt: str = "",
    ):
        self.agent = agent
        self.ctx_list = ctx_list
        self.forget_func = forget_func
        # 初始化消息列表（包含系统提示）
        self.messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
        self.prev_msg_count = 1  # 上一次存储后消息的数量

    def invoke(self, user_input: str) -> str:
        """
        接收用户输入，返回智能体回复，并自动处理存储与遗忘。
        """
        # 构造用户消息

      
        self.messages = self.forget_func(self.messages)
        user_msg = HumanMessage(content=user_input)
        # 构建当前状态（包含历史消息）
        current_state = {"messages": self.messages + [user_msg]}
        # 调用原始 agent
        final_state = self.agent.invoke(current_state)
        # 更新历史消息（含本次交互产生的全部消息）
        self.messages = final_state["messages"]
        # 获取本次新增的消息（即上次存储后到现在的所有新消息）
        new_messages = self.messages[self.prev_msg_count:]
        # 存储新消息（假设 ctx_list.write 接受 BaseMessage 列表）
        self.ctx_list.write(new_messages)

        # 更新下次的起始计数（基于修剪后的消息列表长度）
        self.prev_msg_count = len(self.messages)
        # 返回最后一条消息的内容（即智能体回复）
        return self.messages[-1].content
def create_memory_agent(user_id,agent_id,system_prompt,function_tools,memory_plugin):
    
    ctx_List=ContextList(memory_plugin,agent_id,user_id)
    tools=function_tools+[tool(ctx.read) for ctx in ctx_List.ctx_list]
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=""
    )
    # 包装为自动记忆智能体
    auto_agent = AutoMemoryAgent(
        agent=agent,
        ctx_list=ctx_List,
        forget_func=forget,   # 假设已有 forget 函数
        system_prompt=system_prompt
    )
    return auto_agent