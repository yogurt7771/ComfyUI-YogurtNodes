"""
History Builder Node - 构建和管理会话历史
"""
from typing import List, Tuple, Optional


class HistoryBuilder:
    """
    构建LLM会话历史记录，与现有 generate_text 节点完全兼容
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "previous_history": (
                    "HISTORY",
                    {
                        "tooltip": "之前的会话历史",
                    },
                ),
                "user_message": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "用户消息内容",
                    },
                ),
                "assistant_message": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "助手回复内容",
                    },
                ),
            }
        }

    RETURN_TYPES = ("HISTORY",)
    RETURN_NAMES = ("history",)
    FUNCTION = "build_history"
    OUTPUT_NODE = False

    _NODE_NAME = "History Builder"
    DESCRIPTION = "构建与 LLM 节点兼容的会话历史"
    CATEGORY = "YogurtNodes/LLM"

    def build_history(
        self,
        previous_history: Optional[List[Tuple[str, str]]] = None,
        user_message: str = "",
        assistant_message: str = "",
    ) -> Tuple[List[Tuple[str, str]]]:
        """
        构建会话历史记录，输出格式与现有 LLM 节点兼容
        
        Args:
            previous_history: 之前的历史记录
            user_message: 用户消息
            assistant_message: 助手回复
            
        Returns:
            Tuple containing history list compatible with LLM nodes
        """
        # 如果没有之前的历史，从空列表开始
        if previous_history is None:
            history = []
        else:
            history = list(previous_history)
        
        # 添加新的用户消息
        if user_message.strip():
            history.append(("user", user_message.strip()))
        
        # 添加新的助手回复
        if assistant_message.strip():
            history.append(("assistant", assistant_message.strip()))
        
        return (history,)
