from langchain_core.messages import AIMessage, HumanMessage,ToolMessage,SystemMessage
from qwen_config import llm
def extract_abstract(messages):
    systemprompt=[msg for msg in messages if  isinstance(msg, SystemMessage)]
    initial_state =  messages + [HumanMessage(content="总结上述对话的摘要")]
    abstract= llm.invoke(initial_state)
    print (abstract)
    return systemprompt+[abstract]
    
def forget(messages):
    '''
    遗忘messages中的工具类ToolMessage
    '''
    messages=[msg for msg in messages if not isinstance(msg, ToolMessage)]
    messages=[msg for msg in messages if len(msg.content.strip())>0]
    if len(messages)>2:
        print ("开始提取摘要")
        messages=extract_abstract(messages)
    return messages