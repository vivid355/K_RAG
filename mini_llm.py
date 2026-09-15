"""LLM initialization for mini RAG.

这个文件只负责一件事：创建聊天模型。
以后如果要把 DeepSeek 换成 OpenAI、Ollama 或其他模型，优先改这里。
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from mini_config import BASE_DIR


load_dotenv(BASE_DIR / ".env")


llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    base_url=os.environ.get(
        "DEEPSEEK_BASE_URL",
        "https://api.deepseek.com/v1",
    ),
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
)
