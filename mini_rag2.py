"""Mini Agentic RAG entry point.

现在这个文件只做启动入口。
真正的功能被拆到了：
- mini_retrieval.py：RAG 检索层
- mini_agent_graph.py：单问题 Agent 子图
- mini_main_graph.py：多问题主图
- mini_cli.py：命令行交互
"""

from mini_cli import main


if __name__ == "__main__":
    main()
