"""Command-line interface for mini RAG.

此文件负责把整个项目组装起来并运行
"""

import uuid

from langchain_core.messages import HumanMessage

from mini_agent_graph import create_basic_agent_graph
from mini_logger import (
    log_clarification_needed,
    log_final_answer,
)
from mini_llm import llm
from mini_main_graph import create_main_graph
from mini_retrieval import setup_retrieval


def create_graph_runtime():
    """创建主图main_graph。"""

    tools = setup_retrieval()
    llm_with_tools = llm.bind_tools(tools)
    agent_graph = create_basic_agent_graph(
        llm_with_tools,
        tools,
        llm,
    )
    main_graph = create_main_graph(agent_graph)

    return main_graph


def create_graph_config(thread_id: str) -> dict:
    """把已有的 thread_id 包装成 LangGraph 要求的配置格式。

    接收：
        thread_id：当前会话 ID，来自 CLI 的固定值或 /new 新生成的 uuid。

    输出：
        graph_config：传给 main_graph.invoke/get_state/update_state。

    输出给谁：
        LangGraph checkpointer。它用 thread_id 区分不同多轮对话。
    """

    return {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": 80,
    }


def build_initial_state(question: str) -> dict:
    """首次提问时传入主图的初始状态。

    接收：
        question：用户在 CLI 或 Gradio 输入的原始问题。

    输出：
        dict：MainState 的初始字段，包括 messages/currentQuestion/pendingQuery 等。

    输出给谁：
        main_graph.invoke(...) 的第一个参数，也就是主图 START 节点的输入。

    Bug2 排查意义：
        如果第二轮追问答错，先确认第一轮和第二轮问题是否正确写入
        messages 和 currentQuestion。
    """

    return {
        "messages": [
            HumanMessage(content=question, name="user")
        ],
        "currentQuestion": question,
        "questionIsClear": False,
        "originalQuery": "",
        "pendingQuery": "",
        "rewrittenQuestions": [],
        "clarification_needed": "",
        "agent_answers": [],
        "final_answer": "",
    }


def resume_interrupted_graph(main_graph, graph_config: dict, question: str):
    """当主图因澄清问题暂停时，把用户补充写回图状态并继续运行。

    接收：
        main_graph：已经 compile 的主图，来自 create_graph_runtime()。
        graph_config：包含 thread_id 的会话配置。
        question：用户对澄清问题的补充回答。

    输出：
        main_graph.invoke(None, ...) 的结果。

    输出给谁：
        CLI/Gradio 的结果打印函数或页面渲染函数。

    Bug2 排查意义：
        当系统问“她是谁？”后，用户补充“她是作者”，这里负责把补充写回
        currentQuestion 和 messages，然后让主图从 interrupt 处继续。
    """

    main_graph.update_state(
        graph_config,
        {
            "currentQuestion": question,
            "messages": [
                HumanMessage(content=question, name="user")
            ],
        },
    )

    return main_graph.invoke(
        None,
        config=graph_config,
    )


def print_graph_result(main_graph, graph_config: dict, main_result: dict):
    """根据图是否暂停，打印澄清问题或最终答案。

    接收：
        main_graph：运行后的主图。
        graph_config：当前 thread_id 对应的会话配置。
        main_result：本轮 main_graph.invoke 的返回值。

    输出：
        不返回业务数据，只打印澄清问题或 final_answer。

    输出给谁：
        终端用户。
    """

    snapshot = main_graph.get_state(graph_config)
    values = getattr(snapshot, "values", {}) or {}

    if getattr(snapshot, "next", None):
        log_clarification_needed(
            values.get("clarification_needed", "请补充更完整的问题信息"),
            values.get("pendingQuery", ""),
        )
        return

    log_final_answer(
        main_result.get("final_answer", "")
    )


def main():
    main_graph = create_graph_runtime()

    print("\nTool Calling RAG已就绪，输入问题,/quit退出，/new开启新会话")

    thread_id = "mini_rag2_cli"
    graph_config = create_graph_config(thread_id)

    while True:
        question = input("\n>").strip()

        if question in {"/quit", "/exit"}:
            break

        if not question:
            continue

        if question == "/new":
            thread_id = f"mini_rag2_cli_{uuid.uuid4()}"
            graph_config = create_graph_config(thread_id)
            print("\n====已开启新会话====")
            continue

        snapshot = main_graph.get_state(graph_config)

        if getattr(snapshot, "next", None):
            main_result = resume_interrupted_graph(
                main_graph,
                graph_config,
                question,
            )
        else:
            main_result = main_graph.invoke(
                build_initial_state(question),
                config=graph_config,
            )

        print_graph_result(
            main_graph,
            graph_config,
            main_result,
        )
