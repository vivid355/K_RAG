import operator
from typing import Annotated, Set

from langgraph.graph import MessagesState


def set_union(
    old_keys: Set[str],
    new_keys: Set[str],
) -> Set[str]:
    """Merge set-valued graph state fields."""

    return old_keys | new_keys


def append_unique(
    existing: list[str],
    new: list[str],
) -> list[str]:
    """Append list values while preserving order and removing duplicates."""

    return list(dict.fromkeys(existing + new))


def accumulate_or_reset(
    existing: list[dict],
    new: list[dict],
) -> list[dict]:
    """Merge parallel agent answers, or reset them for a new user turn."""

    if new and any(item.get("__reset__") for item in new):
        return []

    return existing + new


class MainState(MessagesState):
    """主图状态：负责一次完整用户回合的调度数据。

    数据从哪来：
        - CLI/Gradio 首次提问时由 build_initial_state() 创建。
        - 各个主图节点返回 updates 后，由 LangGraph 自动合并进这里。

    输出给谁：
        - main_rewrite_query 读取 currentQuestion/messages/pendingQuery。
        - route_after_main_rewrite 读取 questionIsClear/rewrittenQuestions。
        - aggregate_answers_node 读取 originalQuery/agent_answers。
        - CLI/Gradio 读取 final_answer/clarification_needed。

    Bug2 排查意义：
        第二轮追问答错时，这个类就是你要检查的“主图记忆字段清单”。
    """

    # 当前用户输入。首次来自 build_initial_state；澄清补充来自 update_state。
    currentQuestion: str

    # rewrite_query_simple 的判断结果；决定走澄清还是发给 Agent。
    questionIsClear: bool

    # 较早聊天历史的压缩摘要；给后续追问做长期记忆。
    conversation_summary: str

    # 本轮用于最终聚合的原始问题；可能包含 pendingQuery + 用户补充。
    originalQuery: str

    # 未解决的问题。系统需要澄清时保存，用户补充后再合并改写。
    pendingQuery: str

    # 改写后的独立检索问题列表；route 会把它们 Send 给 Agent 子图。
    rewrittenQuestions: list[str]

    # 系统要问用户补充的信息；CLI/Gradio 会展示它。
    clarification_needed: str

    # 子 Agent 并发返回的答案。使用 reducer，支持多路 Send 结果合并。
    agent_answers: Annotated[list[dict], accumulate_or_reset]

    # 最终给用户看的答案。
    final_answer: str


class AgentState(MessagesState):
    """State for one rewritten question handled by the agent subgraph."""

    question: str
    question_index: int

    iteration_count: Annotated[int, operator.add]
    tool_call_count: Annotated[int, operator.add]
    executed_tool_keys: Annotated[Set[str], set_union]
    retrieval_keys: Annotated[Set[str], set_union]
    retrieved_contexts: Annotated[list[str], append_unique]

    context_summary: str
    agent_answers: list[dict]
