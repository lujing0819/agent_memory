import json
from typing import List, Union

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from qwen_config import llm
# 假设你设置了 OPENAI_API_KEY 环境变量
 

# ========== 1. 定义压缩工具 ==========
@tool
def compress_conversation(conversation_json: str) -> str:
    """
    对话压缩工具，压缩聊天记录，生成一段简洁的摘要。
    输入应为 JSON 格式的字符串，包含一个消息列表，每个消息具有 role 和 content。
    例如：[{"role": "human", "content": "你好"}, {"role": "ai", "content": "你好！"}]
    """
    #try:
    messages = json.loads(conversation_json)
    # except json.JSONDecodeError:
    #     return "错误：输入不是有效的 JSON 格式。"
    print ("aaa")
    # 构建压缩提示词
    prompt = "请将以下对话压缩成一个简洁的摘要，保留关键信息和上下文：\n\n"
    for msg in messages:
        role = "用户" if msg.get("role") == "human" else "助手"
        prompt += f"{role}: {msg.get('content', '')}\n"
    prompt += "\n压缩后的摘要："

    # 调用 LLM 生成摘要
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# ========== 2. 创建智能体 ==========
# 将聊天记录数组转换为工具可接受的 JSON 字符串
def messages_to_json(messages: List[Union[HumanMessage, AIMessage]]) -> str:
    msgs = []
    for msg in messages:
        role = "human" if isinstance(msg, HumanMessage) else "ai"
        msgs.append({"role": role, "content": msg.content})
    return json.dumps(msgs, ensure_ascii=False)

# 智能体工具列表
tools = [compress_conversation]

# 系统提示词：指导智能体如何使用工具
system_prompt = """你是一个对话压缩助手。你的任务是根据用户提供的聊天记录数组，调用 `compress_conversation` 工具生成摘要。
聊天记录会以 JSON 格式传递给你，你需要直接调用工具并返回结果。"""

# 创建智能体（使用 OpenAI Tools 代理）
agent = create_agent(llm, tools, system_prompt=system_prompt)
 

# ========== 3. 示例使用 ==========
if __name__ == "__main__":
    # 模拟聊天记录
    conversation = [
        HumanMessage(content="我今天肚子特别疼"),
        AIMessage(content="吃坏什么东西了"),
        HumanMessage(content="吃的太辣了"),
        AIMessage(content="多喝热水"),
    ]

    # 将聊天记录转换为 JSON 字符串作为智能体输入
    conversation_json = messages_to_json(conversation)

    # 运行智能体
    result = agent.invoke({"input": f"请压缩这段对话：{conversation_json}"})
    print("压缩结果：", result)