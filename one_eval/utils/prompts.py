from __future__ import annotations
from typing import Dict
from pydantic import BaseModel
from one_eval.logger import get_logger
import json

log = get_logger(__name__)



class PromptTemplate(BaseModel):
    """通用 Prompt 模板格式"""
    name: str
    text: str

    def build_prompt(self, **kwargs) -> str:
        return self.text.format(**kwargs)


class PromptRegistry:
    """Prompt 注册中心：全局唯一"""

    def __init__(self):
        self.prompts: Dict[str, PromptTemplate] = {}

    def register(self, name: str, text: str):
        """注册 prompt"""
        self.prompts[name] = PromptTemplate(name=name, text=text)

    def get(self, name: str) -> PromptTemplate:
        if name not in self.prompts:
            log.error(f"[PromptRegistry] 未找到 prompt: {name}")
        return self.prompts[name]


# ----------- 单例实例 -----------
prompt_registry = PromptRegistry()


# ======================================================
# 在下面注册项目所有 prompt
# ======================================================

# -------- Step1: QueryUnderstand Agent --------
prompt_registry.register(
    "query_understand.system",
    """
你是 One-Eval 系统中的 QueryUnderstandAgent。
你的任务是读取用户自然语言输入并输出一个结构化 JSON:
{{
  "is_eval_task": Bool,
  "is_mm": Bool,
  "add_bench_request": Bool,
  "domain": [str, ...],
  "specific_benches": [str, ...],
  "model_path": [str, ...],
  "special_request": str
}}
不要解释，不要添加额外内容，只输出 JSON。
""",
)

prompt_registry.register(
    "query_understand.task",
    """
用户输入如下：

{user_query}

请你根据以上内容严格返回 JSON (必须可被 json.loads 解析):
{{
  "is_eval_task": 是否为评测任务(bool类型),
  "is_mm": 是否涉及多模态任务(bool类型),
  "add_bench_request": 是否需要检索 benchmark(bool类型),
  "domain": ["math", "medical", ...],  # 评测任务的领域，如 ["text", "math", "code"]
  "specific_benches": ["gsm8k", "mmlu", ...],  # 由用户提出的必须评测的指定 benchmark 列表
  "model_path": ["gpt-4o", "local://qwen", ...],  # 被测模型名或本地路径，从用户给的文字描述中寻找，没有则填写 None
  "special_request": "其他无法结构化但依旧重要的需求文本"  # 其他无法结构化但依旧重要的需求,用文字记录用于后续处理
}}
"""
)

prompt_registry.register(
    "bench_search.system",
    """
你是 One-Eval 的 BenchSearchAgent。
你的任务是根据用户需求与本地候选 benchmarks 判断是否需要执行 HF 搜索，并指导工具调用。
最终目标：找到与任务强相关的 benchmark 列表。
"""
)

prompt_registry.register(
    "bench_search.task",
    """
用户需求：
{user_query}

本地匹配到的 benchmarks：
{local_candidates}

若上述数量不足，请调用 hf_search_tool 以：query="用户需求关键词" 搜索更多候选项。
注意：必须让你的输出符合 function-call 规范。
"""
)
