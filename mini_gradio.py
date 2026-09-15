"""Gradio UI for mini Agentic RAG.

这个文件负责把命令行 RAG 包装成一个简单网页聊天界面。
注意：Gradio State 只保存 thread_id 这种简单值，不保存 LangGraph 对象。
"""

import uuid

import gradio as gr

from mini_cli import (
    build_initial_state,
    create_graph_config,
    create_graph_runtime,
    resume_interrupted_graph,
)


MAIN_GRAPH = None


def get_main_graph():
    """懒加载 LangGraph runtime，避免 Gradio 页面初始化阶段就占用 Qdrant。"""

    global MAIN_GRAPH

    if MAIN_GRAPH is None:
        MAIN_GRAPH = create_graph_runtime()

    return MAIN_GRAPH


def create_thread_id() -> str:
    """为一次网页会话创建独立 thread_id。"""

    return f"mini_rag2_gradio_{uuid.uuid4()}"


def answer(message, history, thread_id):
    """处理一次用户输入。"""

    if history is None:
        history = []

    if not thread_id:
        thread_id = create_thread_id()

    message = str(message).strip()

    if not message:
        return "", history, thread_id

    main_graph = get_main_graph()
    graph_config = create_graph_config(thread_id)
    snapshot = main_graph.get_state(graph_config)

    if getattr(snapshot, "next", None):
        main_result = resume_interrupted_graph(
            main_graph,
            graph_config,
            message,
        )
    else:
        initial_state = build_initial_state(message)
        main_result = main_graph.invoke(
            initial_state,
            config=graph_config,
        )

    snapshot = main_graph.get_state(graph_config)
    values = getattr(snapshot, "values", {}) or {}

    if getattr(snapshot, "next", None):
        assistant_message = (
            values.get("clarification_needed", "请补充更完整的问题信息")
            + "\n\n"
            + f"当前 pendingQuery：{values.get('pendingQuery', '')}"
        )
    else:
        assistant_message = main_result.get("final_answer", "")

    # Gradio 6.x Chatbot 默认使用 messages 格式，但不需要传 type="messages"。
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": assistant_message},
    ]

    return "", history, thread_id


def reset_session():
    """重置网页会话。"""

    return [], create_thread_id()


with gr.Blocks(title="Mini Agentic RAG") as demo:
    thread_state = gr.State(create_thread_id())

    gr.Markdown("## Mini Agentic RAG")

    chatbot = gr.Chatbot(
        label="Chat",
        height=520,
    )

    with gr.Row():
        user_input = gr.Textbox(
            placeholder="输入问题，例如：西瓜是谁买的？致爱丽丝是什么？",
            show_label=False,
            scale=8,
        )
        send_btn = gr.Button(
            "发送",
            variant="primary",
            scale=1,
        )

    reset_btn = gr.Button("新会话")

    send_btn.click(
        answer,
        inputs=[user_input, chatbot, thread_state],
        outputs=[user_input, chatbot, thread_state],
    )

    user_input.submit(
        answer,
        inputs=[user_input, chatbot, thread_state],
        outputs=[user_input, chatbot, thread_state],
    )

    reset_btn.click(
        reset_session,
        inputs=[],
        outputs=[chatbot, thread_state],
    )


if __name__ == "__main__":
    demo.launch()
