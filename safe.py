import os
import re
from typing import List, Tuple, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import   HumanMessage  
from qwen_config import safe_llm
 

class SafetyGuard:
    """基于关键词 + 大模型的内容安全防范系统"""

    def __init__(self,
                 safe_llm: ChatOpenAI,
                 sensitive_words: Optional[List[str]] = None,
                 unsafe_categories: Optional[List[str]] = None):
        """
        初始化安全系统
        :param llm: 已配置好的 ChatOpenAI 实例（Qwen 模型）
        :param sensitive_words: 自定义敏感词列表，默认使用内置示例
        :param unsafe_categories: 需要模型检测的不安全内容类别，默认包含“性别对立”、“少儿不宜”等
        """
        self.llm = safe_llm

        # 敏感词列表（可自定义）
        default_sensitive = ["暴力", "色情", "赌博", "毒品", "反动", "辱骂", "诈骗", "恐怖主义"]
        self.sensitive_words = set(sensitive_words) if sensitive_words else set(default_sensitive)
        # 编译正则表达式，用于快速匹配
        self.sensitive_pattern = re.compile('|'.join(re.escape(w) for w in self.sensitive_words)) if self.sensitive_words else None

        # 需要模型检测的类别（用户可指定）
        self.unsafe_categories = unsafe_categories or ["性别对立", "少儿不宜", "暴力", "色情", "歧视"]

    def keyword_filter(self, text: str) -> Tuple[bool, Optional[str]]:
        """敏感词匹配过滤"""
        if not self.sensitive_pattern:
            return True, None
        match = self.sensitive_pattern.search(text)
        if match:
            return False, match.group()
        return True, None

    def model_filter(self, text: str) -> Tuple[bool, str]:
        """
        调用大模型（Qwen）判断文本是否涉及不安全类别
        返回 (是否安全, 原因描述)
        """
        # 构造 prompt，让模型只输出简短判断结果
        categories_str = "、".join(self.unsafe_categories)
        prompt = f"""你是一个内容安全审核助手。请判断以下文本是否涉及以下不安全内容：{categories_str}。如果涉及，请输出“不安全”并说明涉及的具体类别（只输出类别，不输出其他内容）；如果不涉及，请输出“安全”。文本：{text}输出格式：只输出“安全”或“不安全：类别名”，不要有其他内容。"""

        try:
            # 调用模型（使用 langchain 的 invoke 方法）
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = response.content.strip()

            if "不安全" in result:
                # 提取类别（如 "不安全：性别对立"）
                if "：" in result:
                    category = result.split("：")[-1].strip()
                else:
                    category = "未知类别"
                return False, f"模型判定涉及{category}"
            else:
                return True, "模型判定安全"
        except Exception as e:
            # 发生异常时，保守起见返回不安全
            return False, f"模型调用失败: {str(e)}"

    def check(self, text: str) -> Tuple[bool, str]:
        """
        综合安全检查：先关键词过滤，再模型审核
        返回 (是否安全, 原因描述)
        """
        # 第一步：关键词过滤
        safe, matched_word = self.keyword_filter(text)
        if not safe:
            return False, f"包含敏感词: {matched_word}"

        # 第二步：模型审核
        safe, reason = self.model_filter(text)
        if not safe:
            return False, reason

        return True, "内容安全"

    def filter_text(self, text: str, replace_char: str = "*") -> str:
        """将文本中的敏感词替换为指定字符（不改变模型判断）"""
        if not self.sensitive_pattern:
            return text
        return self.sensitive_pattern.sub(lambda m: replace_char * len(m.group()), text)

guard = SafetyGuard(
    safe_llm=safe_llm,
    # 可自定义敏感词和检测类别
    sensitive_words=["暴力", "色情", "赌博", "毒品"],
    unsafe_categories=["性别对立", "少儿不宜", "歧视"]
)
# ========== 使用示例 ==========
if __name__ == "__main__":


    # 2. 创建安全系统实例


    # 3. 测试文本
    test_texts = [
        "这个视频太暴力了，不适合儿童观看。",
        "今天天气真好，适合出去玩。",
        "你这个人怎么这么性别对立，挑起男女矛盾？",
        "有一些少儿不宜的内容，请家长陪同观看。",
        "这是一个正常的新闻内容。",
        "含有反动言论。",
    ]

    for text in test_texts:
        safe, reason = guard.check(text)
        # print(f"文本: {text}")
        # print(f"安全: {safe}, 原因: {reason}")
        # # 展示敏感词替换效果
        # filtered = guard.filter_text(text)
        # if filtered != text:
        #     print(f"过滤后: {filtered}")
        # print("-" * 50)