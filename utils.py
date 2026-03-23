from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage,ToolMessage
def role_content_to_message(role_content):
    role = role_content.get("role")
    content = role_content.get("content")
    name = role_content.get("name", None)
    if role == "assistant":
        return AIMessage(content=content)
    elif role == "user":
        return HumanMessage(content=content)
    elif role == "tool":
        return ToolMessage(content=content, name=name)
    else:
        raise ValueError(f"Unknown role: {role}")


def message_to_role_content(message):
    role_map = {
        "AIMessage": "assistant",
        "HumanMessage": "user",
        "SystemMessage": "system",
        "ToolMessage": "tool", 
        "ChatMessage": "chat",
        "AgentMessage": "agent"
    }
    role=role_map.get(message.__class__.__name__, "unknown")
    if role=="tool":
        return {
        "role": role,
        "content": message.content,
        "name":message.name,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return {
        "role": role,
        "content": message.content,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }