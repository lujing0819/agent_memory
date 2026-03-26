from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import uuid
import os
from pathlib import Path
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any, Dict, List
import json
from collections import deque
from mem0 import Memory
from utils import message_to_role_content,role_content_to_message
from qwen_config import llm,mini_llm
from langchain_core.messages import AIMessage, HumanMessage,ToolMessage
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain.tools import tool
from concurrent.futures import ThreadPoolExecutor
from reranker import text_rerank
os.environ["DASHSCOPE_API_KEY"]=os.getenv("api_key")
class Context(ABC):
    """上下文抽象基类，所有具体上下文必须实现读写方法。"""
    
    def __init__(self, userid: str, agentid: str):
        self.userid = userid
        self.agentid = agentid
        # 可用作上下文的唯一标识
        self.context_id = f"{userid}:{agentid}:{self.__class__.__name__}"
        if not os.path.exists("context"):
            os.makedirs("context")
        self.base_dir = Path("context")
        self.user_agent_dir = self.base_dir / userid / agentid

    def create_context_dirs(self,base_dir: str, userid: str, agentid: str) -> None:
        """
        在 base_dir 下创建 userid/agentid/ 目录，
        并在 agentid 目录下创建四个子目录：history, memory, tool, profile。
        如果目录已存在，不会报错。
        """
        base_path = Path(base_dir) / self.userid / self.agentid
        subdirs = ['history', 'memory', 'tool', 'profile']   
        for sub in subdirs:
            target = base_path / sub
            target.mkdir(parents=True, exist_ok=True)  # 自动创建父目录，存在即忽略
            print(f"确保目录存在: {target}")

    def _get_subdir(self, subname: str) -> Path:
        """获取子目录（如 history），如不存在则自动创建。"""
        subdir = self.user_agent_dir / subname
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir

    def _get_latest_file(self, directory: Path) -> Optional[Path]:
        """返回目录中最后修改时间最新的文件，如果没有则返回 None。"""
        files = [f for f in directory.iterdir() if f.is_file() and "tmp" not in f.name]
 
        if not files:
            return None
        return max(files, key=lambda f: f.stat().st_mtime)

    def _is_within_last_hour(self, file_path: Path) -> bool:
        """判断文件的最后修改时间是否在最近一小时之内。"""
        mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = now - mtime
        return delta.total_seconds() < 3600  # 一小时 = 3600 秒

    def _new_file_path(self, directory: Path, prefix: str = "data") -> Path:
        """生成一个新的文件名，基于当前时间戳。"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return directory / f"{prefix}_{timestamp}.log"
    def _read_lines_from_file(self, file_path: Path, max_lines: Optional[int] = None) -> List[str]:
        """
        从文件中读取行，返回去除换行符的非空行列表。
        若 max_lines 为 None，读取全部行；否则只读取最后 max_lines 行（保持原顺序）。
        """
        lines = []

        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for  line in f.readlines()]
        if max_lines is None or max_lines>len(lines):
            return lines
        else:
            return  lines[-max_lines:]
 


    @abstractmethod
    def read(self, **kwargs) -> Any:
        """读取上下文数据，具体参数由子类定义。"""
        pass
    
    @abstractmethod
    def write(self, **kwargs) -> None:
        """写入上下文数据，具体参数由子类定义。"""
        pass
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} userid={self.userid} agentid={self.agentid}>"

class ContextList:
    def __init__(self,namelist,agentID,userID):
        self.manager = ContextManager()
        self.ctx_list=[ self.manager.get_context(userID,agentID,s) for s in namelist]
        self.namelist=namelist
        self.name=None
    def write(self,messages):
        for ctx in self.ctx_list:
            ctx.write(messages)

class HistoryContext(Context):
    """对话历史上下文，存储消息列表。"""
    
    def __init__(self, userid: str, agentid: str, maxlen: int = 100):
        super().__init__(userid, agentid)
        self.maxlen = maxlen
        self.history_dir = self._get_subdir("history")
        self.name="history"
    def read(self,query ,limit=20, **kwargs) -> List[Dict[str, str]]:
        """
        读取最近的对话历史消息。

        该方法从 history_dir 目录下的文件中按修改时间倒序读取对话记录。
        每个文件内存储了多条消息，每条消息以 JSON 字符串形式存放在单独的行中，
        且文件内的行按时间升序排列（旧消息在前）。方法会从最新的文件开始，
        读取其最后若干行（最多 limit 轮对话对应的消息数），并组装成按时间倒序
        （最新的在前）的消息列表返回。 

        Args:
            limit (int, optional): 需要读取的最近对话轮数（即“对话回合”数），
                每轮对话通常包含两条消息（用户输入和助手回复）。默认值为 2。
            **kwargs: 预留扩展参数，当前未使用。
        """
        print ("读取历史记忆")
        files = [f for f in self.history_dir.iterdir() if f.is_file()]
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        messages = []
        for file_path in files:
            # 读取当前文件最后 remaining 行（文件内升序）
            lines = self._read_lines_from_file(file_path, max_lines=2*limit)
            if not lines:
                continue
            msgs = [eval(line) for line in lines]
            msgs=[ss for s in msgs for ss in s]
            messages=msgs+messages
            if len(messages)/2>=limit:
                break
        messages=[ json.loads(s) for s in set(messages)]  
        messages=[s for s in messages if len(s['content'])>2 and s['role']!='system']     
        messages=sorted(messages,key=lambda s:s['time'],reverse=True)[0:2*limit]
        messages=text_rerank(query,messages,key='content',threshold=0.1)
        t=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result=mini_llm.invoke(f"{messages} 现在的时间是{t}根据上面聊天记录，总结成摘要").content
        return result



    def write(self, messages, **kwargs) -> None:
        """
        写入一条消息，格式如 {"role": "user", "content": "..."}。
        若超过最大长度，自动移除最早的消息。
        """
        latest = self._get_latest_file(self.history_dir)
        if latest and self._is_within_last_hour(latest):
            target_file = latest
        else:
            target_file = self._new_file_path(self.history_dir, prefix="history")
        results=[ message_to_role_content(s) for s in messages]
        results=[ json.dumps(s, ensure_ascii=False) for s in results]
        with open(target_file, "a", encoding="utf-8") as f:
            f.writelines( str(results)+"\n")

class MemoryContext(Context):
    """记忆上下文，存储键值对形式的长期记忆。"""
    
    def __init__(self, userid: str, agentid: str):
        super().__init__(userid, agentid)
        # 建议从环境变量读取API Key，避免硬编码
        DASHSCOPE_API_KEY = "sk-4cf9f15bceea4afda41607e97d7e5db7"
        os.environ["OPENAI_API_KEY"]=DASHSCOPE_API_KEY
        os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.memory_dir =self._get_subdir("memory")
        self.name="memory"
        config = {
            # 嵌入模型部分也需要配置，用于向量化记忆
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "qwen3.5-plus",
                    "temperature": 0.2,
                    "max_tokens": 2000,
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-v3",
                    "embedding_dims": 1024,
                }
            },
            "vector_store": {
                "provider": "faiss",
                "config": {
                    "path":   os.path.abspath(self.memory_dir),      # 索引文件路径
                    "embedding_model_dims": 1024,             # 可选，通常会自动获取
                }
            },
        }

        # 从配置初始化 Mem0
        self.memory=  Memory.from_config(config)
        self.tmp_file =self._get_subdir("memory")/ "tmp.txt"
        
        self.executor = ThreadPoolExecutor(max_workers=8) 
 
    def read(self, query,limit=10, **kwargs) -> Any:
        """
        从记忆存储中检索与 query 语义相关的历史消息。

        该方法使用 mem0 的向量搜索功能，基于当前用户的标识符（self.userid）和查询文本，
        在记忆库中查找最相似的记忆条目，并将结果转换为 LangChain 的 HumanMessage 对象列表，
        便于后续与 LLM 链或代理集成使用。返回的消息按相关性从高到低排序。

        Args:
            query (str): 用于语义搜索的查询字符串，通常为用户当前的输入或需要检索上下文的关键信息。
            limit (int, optional): 最大返回的记忆条目数量。默认值为 10，表示最多返回 10 条相关记忆。
            **kwargs: 预留扩展参数，可传递给底层记忆搜索方法（例如过滤条件、元数据要求等）。
        """
        print ("读取mem0记忆")
        memory_results= self.memory.search(query=query, user_id="123", limit=limit)["results"]
        memory_results=[s['memory'] for s in memory_results if s["score"]<0.7]
        return f"过去的相关记忆是{str(memory_results)}"
    def my_write(self,limit=3) -> None:
        def count_lines(filename):
            """返回文件的行数"""
            with open(filename, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        if count_lines(self.tmp_file) >= limit:   
            with open(self.tmp_file, "r", encoding="utf-8") as f:
                lines=f.readlines()
                lines=[eval(line.strip()) for line in lines if line.strip() and len(line.strip())>0]
                self.memory.add(messages=lines, user_id=self.userid)
            # 清空临时文件
            open(self.tmp_file, "w", encoding="utf-8").close()

    def write(self, messages, **kwargs) -> None: 
        """写入或更新一个键值对。"""
        results=[ message_to_role_content(s) for s in messages]
        results=[json.dumps(s, ensure_ascii=False) for s in results]
        with open(self.tmp_file, "a", encoding="utf-8") as f:
            f.writelines("\n".join(results) + "\n")
        self.executor.submit(self.my_write)
 

class ToolContext(Context):
    """工具调用上下文，记录工具调用历史。"""
    
    def __init__(self, userid: str, agentid: str):
        super().__init__(userid, agentid)
        self._calls: List[Dict[str, Any]] = []
        self.tool_dir = self._get_subdir("tool")
        embedding_model = DashScopeEmbeddings(model="text-embedding-v3")
        self.vector_db = Chroma(persist_directory=str(self.tool_dir/"db") ,embedding_function=embedding_model)
        self.name="tool"
        self.tmp_file =self._get_subdir("tool")/ "tmp.txt"
        self.executor = ThreadPoolExecutor(max_workers=8) 
    def read(self, query) -> Any:
        """
        读取与查询语义相似的工具调用历史记录。
        该方法使用向量数据库进行相似性搜索，从已存储的工具调用记录中检索与给定查询最匹配的条目。
        每个检索到的记录会被封装为 `ToolMessage` 对象，其中包含工具调用的输出、原始查询内容
        以及调用时间（来源于元数据）。目前固定返回最相似的5条记录，`limit` 参数暂未生效，
        未来版本可扩展为动态限制。

        Args:
            query (str): 用于相似性搜索的查询字符串，通常为当前用户输入或需要匹配的关键信息。
            limit (Optional[int], optional): 期望返回的最大记录数量。若为 None，表示返回所有匹配记录。
                注意：当前实现固定返回 k=5 条，该参数暂未使用。保留此参数是为了接口兼容性。
            **kwargs: 预留扩展参数，可传递给向量数据库搜索方法（如过滤条件、搜索类型覆盖等）。
        """
        print ("读取工具记忆")
        docs=self.vector_db.search(query, search_type="similarity", k=5)
        results=[]
        for i,doc in enumerate(docs):
            msg=json.dumps({"id":i,"工具输出":doc.metadata['output']+f"工具调用时间{doc.metadata['time']}","工具输入":doc.page_content},ensure_ascii=False)
            results.append(msg)
        return str(results)      

    def my_write(self,limit=3) -> None:
        def count_lines(filename):
            """返回文件的行数"""
            with open(filename, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        if count_lines(self.tmp_file) >= limit:   
            with open(self.tmp_file, "r", encoding="utf-8") as f:
                lines=f.readlines()
                lines=[json.loads(line.strip()) for line in lines if line.strip() and len(line.strip())>0]
            docs = [Document(page_content=data['page_content'],metadata=data['metadata']) for data in lines]
            self.vector_db.add_documents(docs)
            self.vector_db.persist()
            # 清空临时文件
            open(self.tmp_file, "w", encoding="utf-8").close()

    def write(self, msgs, **kwargs) -> None:
 
        content=[message_to_role_content(msg)['content'] for msg in msgs if isinstance(msg, ToolMessage)]
        if len(content)==0:
            return 
        query=msgs[0].content
        tool_name=msgs[1].additional_kwargs['tool_calls'][0]['function']['name']
        if tool_name.endswith("_read_memory"):
            return
        content=content[0]
        time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data={"page_content":query,"metadata":{"output":content,"time":time,'tool_name':tool_name}}
        with open(self.tmp_file, "a", encoding="utf-8") as f:
            f.writelines(json.dumps(data,ensure_ascii=False) + "\n")
        self.executor.submit(self.my_write)

        


class ProfileContext(Context):
    """用户画像上下文，存储用户属性信息。"""
    
    def __init__(self, userid: str, agentid: str):
        super().__init__(userid, agentid)
        self.tmp_file =self._get_subdir("profile")/ "tmp.txt"
 
        self.executor = ThreadPoolExecutor(max_workers=8) 
        self.profile_dir = self._get_subdir("profile")
        self.name="profile"
 
    def read(self,*args,**kwargs) -> Any:
        """
        读取当前用户的画像信息。
        该方法获取最新的画像文件，
        读取其全部内容，并封装为 LangChain 的 HumanMessage 对象返回。
        画像文件通常为文本格式，记录了用户的偏好、背景、行为模式等结构化或非结构化信息，
        可用于在对话中注入个性化上下文。
        
        """
        print ("读取用户画像")
        latest = self._get_latest_file(self.profile_dir)
        with open(latest, "r", encoding="utf-8") as f:
            profile = f.read()
        return profile.strip()
    
    def my_write(self,limit=5) -> None:
        def count_lines(filename):
            """返回文件的行数"""
            with open(filename, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        if count_lines(self.tmp_file) >= limit:   
 
            with open(self.tmp_file, "r", encoding="utf-8") as f:
                lines=f.readlines()
            lines=[json.loads(line.strip())['content'] for line in lines if len(line.strip())>0]
            content="\n".join(lines)
            profile=llm.invoke(f"{content} 根据上述内容，总结出用户画像").content
            latest = self._get_latest_file(self.profile_dir)
            if latest and self._is_within_last_hour(latest):
                target_file = latest
            else:
                target_file = self._new_file_path(self.profile_dir, prefix="profile")
            with open(target_file,"w",encoding="utf-8") as f:
                f.writelines(profile)

            # 清空临时文件
            open(self.tmp_file, "w", encoding="utf-8").close()

    def write(self, messages, **kwargs) -> None: 
        """写入或更新一个键值对。"""
 
        results=[ message_to_role_content(s) for s in messages]
        results=[ json.dumps(s,ensure_ascii=False) for s in results if s['role']=="user"]
        with open(self.tmp_file, "a", encoding="utf-8") as f:
            f.writelines("\n".join(results) + "\n")
        self.executor.submit(self.my_write)


class DocumentContext(Context):
    """用户画像上下文，存储用户属性信息。"""
    
    def __init__(self, userid: str, agentid: str):
        super().__init__(userid, agentid)
        persist_directory =str(self._get_subdir("documents")/ "vector_db")
        embeddings = DashScopeEmbeddings(model="text-embedding-v3")
        self.vector_db = Chroma(persist_directory=persist_directory,embedding_function=embeddings)
    def read(self, query,**kwargs) -> Any:
        """
        从用户的个人知识库中检索与查询最相关的信息。
        当问到专业知识，个人知识时，调用
        Args:
            query (str): 用于相似性搜索的查询字符串，通常为当前用户输入或需要匹配的关键信息。
        """
        print ("读取个人知识库")
        docs=self.vector_db.search(query,search_type="similarity",k=5)
        docs=[s.metadata["content"] for s in docs]
        results=text_rerank(query,docs,threshold=0.75)
        return str(results)
    def write(self, messages, **kwargs) -> None: 
        return 



# ========== 可选的管理类，用于统一获取上下文实例 ==========
class ContextManager:
    """上下文管理器，负责创建和缓存上下文实例。"""
    
    def __init__(self):
        self._contexts: Dict[str, Context] = {}
    def get_context(self, userid: str, agentid: str, context_type: str) -> Context:
        """
        根据用户ID、代理ID和类型获取上下文实例。
        若不存在则创建新实例（此处简化，均创建新实例，实际可缓存）。
        """
        # 简单工厂模式
        context_class = {
            "history": HistoryContext,
            "memory": MemoryContext,
            "tool": ToolContext,
            "profile": ProfileContext,
            "document":DocumentContext
        }.get(context_type.lower())
        
        if not context_class:
            raise ValueError(f"Unknown context type: {context_type}")
        
        # 生成唯一键用于缓存（可选）
        key = f"{userid}:{agentid}:{context_type}"
        if key not in self._contexts:
            self._contexts[key] = context_class(userid, agentid)
        return self._contexts[key]
if __name__ == "__main__":
    agent_id="agent_001"
    user_id="user_123"
    ctx=DocumentContext("user_123","agent_001")
    result=ctx.read("小孩有些多动")
    print (result)