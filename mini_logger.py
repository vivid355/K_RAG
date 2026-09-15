"""Lightweight execution logger for mini RAG.

统一管理节点、路由、工具调用、上下文检查和最终输出日志。
以后如果要接 UI、文件日志、LangSmith，优先改这里。
"""

from pprint import pformat


def _short_text(value, max_len: int = 500) -> str:
    """截断过长日志，避免终端刷屏。"""

    text = str(value)

    if len(text) <= max_len:
        return text

    return text[:max_len] + "...[truncated]"


def log_node(node_name: str):
    """打印进入的 LangGraph 节点。"""

    print(f"\n[节点] {node_name}")


def log_main_node(node_name: str):
    """打印主图节点。"""

    print(f"\n[主图节点] {node_name}")


def log_route(from_node: str, to_node):
    """打印路由跳转。to_node 可以是字符串，也可以是 Send 列表。"""

    if isinstance(to_node, list):
        print(f"[路由] {from_node} -> Send({len(to_node)} tasks)")
    else:
        print(f"[路由] {from_node} -> {to_node}")


def log_budget(
    iteration_count: int,
    max_iterations: int,
    tool_call_count: int,
    max_tool_calls: int,
):
    """打印 Agent 预算。"""

    print(
        "[预算] "
        f"LLM迭代={iteration_count}/{max_iterations}, "
        f"工具请求={tool_call_count}/{max_tool_calls}"
    )


def log_context_check(token_count: int, threshold: int):
    """打印上下文 token 检查。"""

    print(
        "[上下文检查] "
        f"Token估算={token_count}, "
        f"压缩阈值={threshold}"
    )


def log_tool_call(tool_name: str, tool_args: dict, tool_key: str):
    """打印工具调用。"""

    print(f"[工具调用] {tool_name}")
    print(f"key: {tool_key}")
    print("args:")
    print(pformat(tool_args, width=100, compact=True))


def log_tool_result(tool_name: str, result, max_len: int = 500):
    """打印工具结果摘要。"""

    print(f"[工具结果] {tool_name}")
    print(_short_text(result, max_len=max_len))


def log_llm_response(content, tool_calls=None):
    """打印 LLM 返回内容和 tool calls。"""

    print("content:", _short_text(content, max_len=500))
    print("tool_calls:", tool_calls or [])


def log_rewrite_result(
    pending_query: str,
    current_question: str,
    is_clear: bool,
    questions: list,
    clarification_needed: str,
):
    """打印 query rewrite 结构化结果。"""

    print("\n====查询改写结果====")
    print("pending_query:", pending_query)
    print("current_question:", current_question)
    print("is_clear:", is_clear)
    print("questions:", questions)
    print("clarification_needed:", clarification_needed)


def log_history_summary(summary: str):
    """打印历史摘要长度。"""

    print(f"[历史摘要]长度={len(summary)}")


def log_context_summary(summary: str):
    """打印检索上下文压缩结果。"""

    print(f"[压缩完成]摘要长度={len(summary)}")


def log_final_answer(answer: str):
    """打印最终答案。"""

    print("\n====最终回答====")
    print(answer)


def log_clarification_needed(clarification: str, pending_query: str):
    """打印澄清提示。"""

    print("\n====需要补充信息====")
    print(clarification)
    print("\n====图已暂停，当前pendingQuery=====")
    print(pending_query)
