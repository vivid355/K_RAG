# K_RAG

一个用于学习 Agentic RAG 的小型项目，包含父子分块、Qdrant 混合检索、LangGraph 工具调用、多问题拆分、澄清中断与多轮对话记忆。

## 安装

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

复制环境变量模板并填写自己的 DeepSeek API Key：

```bash
copy .env.example .env
```

将 Markdown 文档放入 `markdown_docs/`，然后运行命令行版本：

```bash
python mini_rag2.py
```

或者运行 Gradio 界面：

```bash
python mini_gradio.py
```

本地 `.env`、文档、Qdrant 索引、父块存储、缓存和个人笔记均已加入 `.gitignore`，不会提交到仓库。
