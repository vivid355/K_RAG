"""Single-question Agent graph for mini_rag2.

这个文件负责“一个子问题怎么被 Agent 回答”：
1. orchestrator 调用带 tools 的 LLM。
2. tools 节点执行检索工具。
3. 上下文太长时压缩检索证据。
4. 工具预算用完时生成兜底答案。
5. collect_answer 把子答案交回主图。
"""

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph

from mini_config import (
    BASE_TOKEN_THRESHOLD,
    MAX_ITERATIONS,
    MAX_TOOL_CALLS,
    TOKEN_GROWTH_FACTOR,
)
from mini_logger import (
    log_budget,
    log_context_check,
    log_context_summary,
    log_llm_response,
    log_node,
    log_route,
    log_tool_call,
    log_tool_result,
)
from mini_prompts import get_agent_system_prompt
from mini_state import AgentState
from mini_utils import (
    estimate_context_tokens,
    extract_retrieval_contexts,
    make_tool_call_key,
)


def build_agent_initial_messages(rewritten_question: str) -> list:
    """为单问题 Agent 创建初始 system + human 消息。"""

    return [
        SystemMessage(content=get_agent_system_prompt()),
        HumanMessage(content=rewritten_question),
    ]


def create_basic_agent_graph(llm_with_tools, tools, plain_llm):
    """创建单问题 Agent 子图。"""

    tools_by_name = {
        current_tool.name: current_tool for current_tool in tools
    }

    def orchestrator(state: AgentState):
        log_node("agent.orchestrator")

        graph_messages = state["messages"]
        context_summary = state.get(
            "context_summary",
            "",
        ).strip()

        if context_summary:
            invoke_messages = [
                graph_messages[0],
                HumanMessage(
                    content=(
                        "[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n"
                        f"{context_summary}"
                    )
                ),
                *graph_messages[1:],
            ]
        else:
            invoke_messages = graph_messages

        response = llm_with_tools.invoke(
            invoke_messages
        )
        current_tool_calls = response.tool_calls or []

        log_llm_response(response.content, current_tool_calls)

        return {
            "messages": [response],
            "iteration_count": 1,
            "tool_call_count": len(current_tool_calls),
        }

    def execute_tools(state: AgentState):
        log_node("agent.tools")

        last_message = state["messages"][-1]
        tool_calls = getattr(
            last_message,
            "tool_calls",
            None,
        ) or []
        old_keys = set(
            state.get("executed_tool_keys", set())
        )
        new_keys = set()
        tool_messages = []

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]
            tool_key = make_tool_call_key(
                tool_name,
                tool_args,
            )

            log_tool_call(tool_name, tool_args, tool_key)

            if tool_key in old_keys or tool_key in new_keys:
                tool_result = (
                    f"DUPLICATE_TOOL_CALL:{tool_key}已经执行过"
                    "请使用之前的结果，如果证据不足"
                    "请使用不同的查询词"
                )
                log_tool_result(tool_name, tool_result)
            else:
                selected_tool = tools_by_name.get(
                    tool_name
                )

                if selected_tool is None:
                    tool_result = (
                        f"UNKNOWN_TOOL:{tool_name}"
                    )
                    log_tool_result(tool_name, tool_result)
                else:
                    tool_result = selected_tool.invoke(tool_args)
                    log_tool_result(tool_name, tool_result)
                    new_keys.add(tool_key)

            tool_messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            )

        return {
            "messages": tool_messages,
            "executed_tool_keys": new_keys,
            "retrieval_keys": new_keys,
        }

    def route_after_tools(state: AgentState):
        tool_messages = [
            message for message in state["messages"]
            if isinstance(message, ToolMessage)
        ]
        context_summary = state.get(
            "context_summary",
            "",
        ).strip()

        contents_to_measure = list(tool_messages)

        if context_summary:
            contents_to_measure.append(
                HumanMessage(content=context_summary)
            )

        token_count = estimate_context_tokens(
            contents_to_measure
        )
        iteration_count = state.get(
            "iteration_count",
            0,
        )
        threshold = int(
            BASE_TOKEN_THRESHOLD
            * (
                1 + TOKEN_GROWTH_FACTOR * max(iteration_count - 1, 0)
            )
        )

        log_context_check(token_count, threshold)

        if token_count >= threshold:
            log_route("agent.tools", "compress_context")
            return "compress"

        log_route("agent.tools", "orchestrator")
        return "continue"

    def compress_context(state: AgentState):
        log_node("agent.compress_context")

        existing_summary = state.get(
            "context_summary",
            "",
        ).strip()
        tool_contents = [
            str(message.content) for message in state["messages"]
            if isinstance(message, ToolMessage) and str(message.content).strip()
        ]
        compression_input = []

        if existing_summary:
            compression_input.append(
                "之前的研究摘要：\n"
                f"{existing_summary}"
            )

        if tool_contents:
            compression_input.append(
                "本轮新增检索结果:\n"
                + "\n\n---\n\n".join(tool_contents)
            )

        evidence = "\n\n".join(compression_input)
        response = plain_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "你是RAG研究上下文压缩器。"
                        "请保留与用户问题直接相关的事实"
                        "人物，时间，数字和来源文件名"
                        "删除重复内容，工具调用信息"
                        "parent_id和无关片段"
                        "不能添加原文件不存在的事实"
                        "只返回压缩后的研究摘要"
                    )
                ),
                HumanMessage(
                    content=(
                        f"用户问题:\n{state['question']}\n\n"
                        f"需要压缩的证据:\n{evidence}"
                    )
                ),
            ]
        )
        summary = response.content.strip()

        log_context_summary(summary)

        removals = [
            RemoveMessage(id=message.id)
            for message in state["messages"]
            if (
                isinstance(message, (AIMessage, ToolMessage))
                and getattr(message, "id", None)
            )
        ]

        return {
            "messages": removals,
            "context_summary": summary,
        }

    def route_after_orchestrator(state: AgentState):
        last_message = state["messages"][-1]
        tool_calls = getattr(
            last_message,
            "tool_calls",
            None,
        ) or []
        iteration_count = state.get(
            "iteration_count",
            0,
        )
        tool_call_count = state.get(
            "tool_call_count",
            0,
        )

        log_budget(
            iteration_count,
            MAX_ITERATIONS,
            tool_call_count,
            MAX_TOOL_CALLS,
        )

        if not tool_calls:
            log_route("agent.orchestrator", "collect_answer")
            return "collect_answer"

        if (
            iteration_count >= MAX_ITERATIONS
            or tool_call_count >= MAX_TOOL_CALLS
        ):
            log_route("agent.orchestrator", "fallback_response")
            return "fallback"

        log_route("agent.orchestrator", "tools")
        return "tools"

    def fallback_response(state: AgentState):
        log_node("agent.fallback_response")

        retrieved_contents = []

        for message in state["messages"]:
            if isinstance(message, ToolMessage):
                content = str(message.content).strip()

                if content:
                    retrieved_contents.append(content)

        context_summary = state.get(
            "context_summary",
            "",
        ).strip()

        if context_summary:
            retrieved_contents.append(
                "压缩后的研究证据:\n"
                f"{context_summary}"
            )

        if retrieved_contents:
            evidence = "\n\n---\n\n".join(
                retrieved_contents
            )
        else:
            evidence = "没有已经成功取得的文档证据"

        response = plain_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "你是文档问答助手。"
                        "工具调用预算已经结束。"
                        "只能根据下面已经取得的文档证据回答，"
                        "不能使用外部知识或编造内容。"
                        "证据不足时明确指出不足。"
                        "来源使用格式：\n"
                        "Sources:\n"
                        "- 文件名"
                    )
                ),
                HumanMessage(
                    content=(
                        f"用户问题:\n{state['question']}\n\n"
                        f"已经取得的文档证据:\n{evidence}"
                    )
                ),
            ]
        )

        return {
            "messages": [response]
        }

    def collect_answer(state: AgentState):
        log_node("agent.collect_answer")

        last_message = state["messages"][-1]

        if (
            isinstance(last_message, AIMessage)
            and last_message.content
            and not getattr(last_message, "tool_calls", None)
        ):
            answer = last_message.content
        else:
            answer = "无法生成有效答案"

        question_index = state.get("question_index", 0)
        question = state.get("question", "")
        retrieved_contexts = extract_retrieval_contexts(
            state.get("messages", [])
        )
        fallback_summary = str(
            state.get("context_summary", "")
        ).strip()

        if not retrieved_contexts and fallback_summary:
            retrieved_contexts = [fallback_summary]

        return {
            "retrieved_contexts": retrieved_contexts,
            "agent_answers": [
                {
                    "index": question_index,
                    "question": question,
                    "answer": answer,
                    "contexts": retrieved_contexts,
                }
            ],
        }

    builder = StateGraph(AgentState)
    builder.add_node(
        "orchestrator",
        orchestrator,
    )
    builder.add_node(
        "tools",
        execute_tools,
    )
    builder.add_node(
        "fallback_response",
        fallback_response,
    )
    builder.add_node(
        "compress_context",
        compress_context,
    )
    builder.add_node(
        "collect_answer",
        collect_answer,
    )
    builder.add_edge(
        START,
        "orchestrator",
    )
    builder.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "tools": "tools",
            "fallback": "fallback_response",
            "collect_answer": "collect_answer",
        },
    )
    builder.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "compress": "compress_context",
            "continue": "orchestrator",
        },
    )
    builder.add_edge(
        "compress_context",
        "orchestrator",
    )
    builder.add_edge(
        "fallback_response",
        "collect_answer",
    )
    builder.add_edge(
        "collect_answer",
        END,
    )

    return builder.compile()
