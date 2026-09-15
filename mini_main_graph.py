"""Main graph for multi-question RAG orchestration.

此文件是主图全流程
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from mini_logger import (
    log_history_summary,
    log_main_node,
    log_rewrite_result,
    log_route,
)

from mini_agent_graph import build_agent_initial_messages
from mini_config import MAIN_HISTORY_MESSAGES_TO_KEEP
from mini_llm import llm
from mini_prompts import (
    get_conversation_summary_prompt,
    get_rewrite_query_prompt,
)
from mini_schemas import QueryAnalysis
from mini_state import MainState
from mini_utils import (
    format_conversation,
    is_plain_conversation_message,
    remove_messages_not_in,
)


def rewrite_query_simple(
    question: str,
    pending_query: str = "",
    conversation_summary: str = "",
    recent_conversation: str = "",
) -> QueryAnalysis:
    """判断问题是否清晰，并把清晰问题改写成适合检索的子问题列表。

    接收：
        question：当前用户问题，来自 MainState["currentQuestion"]。
        pending_query：上一轮未解决的问题，来自 MainState["pendingQuery"]。
        conversation_summary：较早历史对话摘要，来自 MainState["conversation_summary"]。
        recent_conversation：最近几轮原始对话，由 main_rewrite_query 现场整理。

    输出：
        QueryAnalysis：
        - is_clear：问题是否清晰。
        - questions：清晰时给子 Agent 检索用的独立问题列表。
        - clarification_needed：不清晰时要问用户补充什么。

    输出给谁：
        main_rewrite_query 节点。它根据 QueryAnalysis 决定：
        - 进入 request_clarification；
        - 或 Send 多个子问题给 agent 子图。

    Bug2 排查意义：
        “第一次答对，第二次追问答错”时，优先看这里打印的 questions。
        如果 questions 已经把“她”改错了，后面检索和生成都会沿着错误问题执行。
    """

    llm_with_structure = llm.with_structured_output(
        QueryAnalysis,
        method="function_calling",
    )
    context_parts = []

    if conversation_summary:
        context_parts.append(
            "历史对话摘要:\n"
            f"{conversation_summary}"
        )

    if recent_conversation:
        context_parts.append(
            "最近对话:\n"
            f"{recent_conversation}"
        )

    if pending_query:
        context_parts.append(
            "之前未解决的问题:\n"
            f"{pending_query}\n\n"
            "用户补充:\n"
            f"{question}\n\n"
            "请把“之前未解决的问题”和“用户补充”合并成一个清晰，完整，适合检索的问题"
        )
    else:
        context_parts.append(
            "当前用户问题：\n"
            f"{question}"
        )

    user_content = "\n\n".join(context_parts)
    response = llm_with_structure.invoke(
        [
            SystemMessage(content=get_rewrite_query_prompt()),
            HumanMessage(content=user_content),
        ]
    )

    # 兜底：模型判断清晰但忘记返回 questions 时，至少保留原问题。
    if response.is_clear and not response.questions:
        if pending_query:
            fallback_question = (
                f"{pending_query}\n"
                f"用户补充:{question}"
            )
        else:
            fallback_question = question

        response = response.model_copy(
            update={"questions": [fallback_question]}
        )

    log_rewrite_result(
        pending_query=pending_query,
        current_question=question,
        is_clear=response.is_clear,
        questions=response.questions,
        clarification_needed=response.clarification_needed,
    )

    return response


def aggregate_simple(
    original_question: str,
    agent_answers: list[dict],
) -> str:
    """把多个子 Agent 的答案汇总成最终回答。

    接收：
        original_question：用户原始问题或 pendingQuery + 用户补充。
        agent_answers：多个子 Agent 返回的结构化答案，来自 MainState["agent_answers"]。

    输出：
        final_answer 字符串。

    输出给谁：
        aggregate_answers_node，再写入 MainState["final_answer"] 和 messages。

    Bug2 排查意义：
        如果 rewrite 和检索都正确，但最终答案仍然错，才怀疑这里的聚合阶段。
    """

    if not agent_answers:
        return "没有生成任何子问题答案"

    sorted_answers = sorted(
        agent_answers,
        key=lambda item: item.get("index", 0),
    )
    formatted_answers = []

    for item in sorted_answers:
        formatted_answers.append(
            f"子问题{item.get('index', 0) + 1}:{item.get('question', '')}\n"
            f"子答案{item.get('index', 0) + 1}:\n{item.get('answer', '')}"
        )

    answers_text = "\n\n---\n\n".join(formatted_answers)
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "你是 RAG 最终答案汇总助手。"
                    "你只能根据给定的子问题答案进行汇总，不能加入外部知识。"
                    "如果子答案中信息不足，要明确说明不足。"
                    "不要提到内部流程、Agent、工具调用或检索过程。"
                    "保留重要的人名、时间、事件、数字和文件来源。"
                    "如果多个子答案里有 Sources，请在最终回答末尾合并去重。"
                    "来源格式必须是：\n"
                    "Sources:\n"
                    "- 文件名"
                )
            ),
            HumanMessage(
                content=(
                    f"用户原始问题:\n{original_question}\n\n"
                    f"多个子问题答案:\n{answers_text}\n\n"
                    "请汇总成一个自然，完整，简洁的最终回答"
                )
            ),
        ]
    )

    return response.content


def create_main_graph(agent_graph):
    """创建主图：负责澄清、多问题调度和最终聚合。

    接收：
        agent_graph：单问题 Agent 子图，来自 mini_agent_graph.create_basic_agent_graph。

    输出：
        compile 后的 main_graph。

    输出给谁：
        CLI/Gradio runtime。用户每次提问最终都会进入这个主图。

    主图职责：
        用户问题 -> 历史摘要 -> 查询改写 -> 澄清或发送子 Agent -> 聚合答案。
    """
#2
    def summarize_history_node(state: MainState):
        """压缩较早历史对话，只保留追问需要的摘要。

        接收：
            state：LangGraph 当前 MainState，主要读取 messages/conversation_summary。

        输出：
            updates dict，可能包含：
            - conversation_summary：新的历史摘要。
            - messages：RemoveMessage，用来删除过早的原始消息。
            - agent_answers reset：清空上一轮子答案。

        输出给谁：
            LangGraph 状态合并器，然后流向 main_rewrite_query。

        Bug2 排查意义：
            如果第二轮追问依赖很早之前的信息，要看摘要有没有保留关键对象。
        """
        log_main_node("summarize_history")

        messages = state.get("messages", [])#读取state里的messages(Humanmessage+AIMessage)
        existing_summary = state.get("conversation_summary", "").strip()#读取历史对话摘要
        updates = {
            "agent_answers": [{"__reset__": True}],
        }

        if not messages:
            return updates

        plain_messages = [
            message for message in messages
            if is_plain_conversation_message(message)
        ]
        keep_count = MAIN_HISTORY_MESSAGES_TO_KEEP

        if len(plain_messages) > keep_count:
            messages_to_summarize = plain_messages[:-keep_count]
            messages_to_keep = plain_messages[-keep_count:]
        else:
            messages_to_summarize = []
            messages_to_keep = plain_messages

        keep_ids = {
            getattr(message, "id", None) for message in messages_to_keep
        }
        keep_ids.discard(None)
        removals = remove_messages_not_in(messages, keep_ids)

        if removals:
            updates["messages"] = removals

        if not messages_to_summarize:
            return updates

        summary_input = (
            "已有历史摘要:\n"
            f"{existing_summary or '无'}\n\n"
            "需要合并进摘要的较早对话:\n"
            f"{format_conversation(messages_to_summarize)}"
        )
        response = llm.invoke(
            [
                SystemMessage(content=get_conversation_summary_prompt()),
                HumanMessage(content=summary_input),
            ]
        )
        updates["conversation_summary"] = response.content.strip()
        log_history_summary(updates["conversation_summary"])

        return updates
#6
    def main_rewrite_query(state: MainState):
        """主图的查询改写节点，把当前用户问题变成可检索的独立问题。

        接收：
            state：MainState，重点读取：
            - currentQuestion：本轮用户输入。
            - pendingQuery：上一轮未澄清的问题。
            - conversation_summary：较早历史摘要。
            - messages：最近原始对话。

        输出：
            如果问题不清晰：
                写入 questionIsClear=False、pendingQuery、clarification_needed。
            如果问题清晰：
                写入 questionIsClear=True、originalQuery、rewrittenQuestions。

        输出给谁：
            route_after_main_rewrite。它根据 questionIsClear 决定走澄清还是 Agent。

        Bug2 排查意义：
            第二轮追问答错，80% 先看这个节点整理出的 recent_conversation
            和 rewrite_query_simple 返回的 rewrittenQuestions。
        """
        log_main_node("main_rewrite_query")

        current_question = str(
            state.get("currentQuestion", "")
        ).strip()
        pending_query = str(
            state.get("pendingQuery", "")
        ).strip()
        conversation_summary = str(
            state.get("conversation_summary", "")
        ).strip()
        plain_messages = [
            message for message in state.get("messages", [])
            if is_plain_conversation_message(message)
        ]
        recent_messages = plain_messages[:-1]
        recent_conversation = format_conversation(
            recent_messages[-MAIN_HISTORY_MESSAGES_TO_KEEP:]
        )
        analysis = rewrite_query_simple(#调用rewrite_query_simple的评判AI
            current_question,
            pending_query=pending_query,
            conversation_summary=conversation_summary,
            recent_conversation=recent_conversation,
        )

        if not analysis.is_clear:
            clarification = (
                analysis.clarification_needed.strip()
                or "请补充更完整的问题信息"
            )

            if pending_query:
                next_pending_query = (
                    f"{pending_query}\n"
                    f"用户补充：{current_question}"
                )
            else:
                next_pending_query = current_question

            return {
                "questionIsClear": False,
                "originalQuery": "",
                "pendingQuery": next_pending_query,
                "rewrittenQuestions": [],
                "clarification_needed": clarification,
                "agent_answers": [{"__reset__": True}],
                "final_answer": "",
            }

        rewritten_questions = analysis.questions or [current_question]

        if pending_query:
            original_query = (
                f"{pending_query}\n"
                f"用户补充:{current_question}"
            )
        else:
            original_query = current_question

        return {
            "questionIsClear": True,
            "originalQuery": original_query,
            "pendingQuery": "",
            "rewrittenQuestions": rewritten_questions,
            "clarification_needed": "",
            "agent_answers": [{"__reset__": True}],
            "final_answer": "",
        }
#3
    def route_after_main_rewrite(state: MainState):
        """根据 query rewrite 结果决定主图下一步走向。

        接收：
            state：main_rewrite_query 写回后的 MainState。

        输出：
            - "request_clarification"：问题不清楚，暂停等用户补充。
            - "aggregate_answers"：没有可检索子问题。
            - Send(...) 列表：把多个 rewrittenQuestions 并发发给 agent 子图。

        输出给谁：
            LangGraph 条件边。

        Bug2 排查意义：
            如果系统该澄清却没澄清，或者该发 Agent 却没发，要看这里的路由。
        """
        if not state.get("questionIsClear", False):
            log_route("main.rewrite_query","request_clarification")
            return "request_clarification"

        rewritten_questions = state.get("rewrittenQuestions", [])

        if not rewritten_questions:
            log_route("main.rewrite_query","aggregate_answers")
            return "aggregate_answers"

        sends=[
            Send(
                "agent",
                {
                    "question":rewritten_question,
                    "question_index":index,
                    "messages":build_agent_initial_messages(
                        rewritten_question
                    ),
                    "iteration_count":0,
                    "tool_call_count":0,
                    "executed_tool_keys":set(),
                    "retrieval_keys":set(),
                    "retrieved_contexts":[],
                    "context_summary":"",
                    "agent_answers":[],
                },
            )
            for index,rewritten_question in enumerate(rewritten_questions)
        ]
        log_route("main.rewrite_query",sends)
        return sends
#4
    def request_clarification_node(state: MainState):
        """澄清暂停节点。

        接收：
            state：包含 clarification_needed 和 pendingQuery 的 MainState。

        输出：
            空 dict。真正的暂停由 compile(... interrupt_before=[...]) 控制。

        输出给谁：
            LangGraph interrupt 机制。CLI/Gradio 读取 state 后展示澄清问题。
        """
        log_main_node("request_clarification")
        return {}
#5
    def aggregate_answers_node(state: MainState):
        """最终聚合节点，把子 Agent 答案合成用户看到的最终回答。

        接收：
            state：包含 originalQuery 和 agent_answers 的 MainState。

        输出：
            - final_answer：最终答案。
            - messages：AIMessage，用于进入多轮历史。

        输出给谁：
            CLI/Gradio 展示；同时 messages 会成为后续追问的历史上下文。

        Bug2 排查意义：
            这个节点写入的 AIMessage 会进入下一轮 recent_conversation。
            如果助手上一轮回答不清楚，第二轮 rewrite 也可能受影响。
        """
        log_main_node("aggregate_answers")

        final_answer = aggregate_simple(
            original_question=state.get("originalQuery", ""),
            agent_answers=state.get("agent_answers", []),
        )

        return {
            "final_answer": final_answer,
            "messages": [
                AIMessage(content=final_answer, name="assistant")
            ],
        }
#1
    builder = StateGraph(MainState)
    builder.add_node("summarize_history", summarize_history_node)
    builder.add_node("main_rewrite_query", main_rewrite_query)
    builder.add_node("request_clarification", request_clarification_node)
    builder.add_node("agent", agent_graph)
    builder.add_node("aggregate_answers", aggregate_answers_node)
    builder.add_edge(START, "summarize_history")
    builder.add_edge("summarize_history", "main_rewrite_query")
    builder.add_conditional_edges(
        "main_rewrite_query",
        route_after_main_rewrite,
    )
    builder.add_edge("request_clarification", "main_rewrite_query")
    builder.add_edge(["agent"], "aggregate_answers")
    builder.add_edge("aggregate_answers", END)

    checkpointer = InMemorySaver()

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["request_clarification"],
    )
