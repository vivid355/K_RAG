def get_rewrite_query_prompt() -> str:
    """返回 Query Rewrite 节点使用的 System Prompt。

    接收：
        无参数。这里只返回固定规则文本。

    输出：
        str：发给 LLM 的 SystemMessage 内容。

    输出给谁：
        mini_main_graph.rewrite_query_simple()。

    注意：
        用户当前问题、recent_conversation、conversation_summary、pendingQuery
        不在这里拼接，而是在 rewrite_query_simple() 里作为 HumanMessage 拼接。

    Bug2 排查意义：
        如果“她/他/它/这个/那个”等追问被改写错，要看这里有没有足够强地要求：
        - 优先用历史对话解析指代；
        - 不能确定就要求澄清；
        - 不要乱猜。
    """

    return """你是RAG系统中的查询改写助手
    你的任务：
1. 判断用户问题是否清晰。
2. 如果清晰，把它改写成适合向量数据库检索的独立问题。
3. 如果问题里有“他、她、它、这个、那个、上面、刚才的”等模糊指代，请优先根据历史对话摘要和最近对话解析。
4. 如果仍然无法判断指代对象，再标记为不清晰。
5. 如果输入中包含“之前未解决的问题”和“用户补充”，请把两者合并成一个清晰、完整、适合检索的问题。
6. 不要编造用户没说的信息。
7. 保留人名、时间、地点、文件名、专有名词和数字。
8. 如果用户问的是一个明确事实问题，通常应该认为它是清晰的。
9. 如果问题包含多个独立问题，最多拆成 3 个。

输出要求：
- is_clear: 问题是否清晰。
- questions: 清晰时，放改写后的问题列表。
- clarification_needed: 不清晰时，说明需要用户补充什么。
"""


def get_conversation_summary_prompt() -> str:
    """Prompt used to compress older chat history."""

    return """
你是 RAG 系统的对话历史压缩器。

你的任务：
1. 把较早的用户/助手对话合并成一个简短摘要。
2. 保留后续追问可能需要的上下文，比如人物、物品、时间、事件、来源文件名。
3. 删除寒暄、重复内容、工具调用细节和无关内容。
4. 不要添加原对话没有的信息。
5. 控制在 50～120 字。

只返回摘要文本，不要加标题。
如果没有有用信息，返回空字符串。

"""


def get_agent_system_prompt() -> str:
    """Prompt used by each single-question agent subgraph."""

    return """你是文档问答助手，只能根据工具返回的文档回答
    规则：
1. 回答问题前必须先调用 search_child_chunks。
2. 当短文本相关但上下文不完整时，调用 retrieve_parent_chunks。
3. parent_id 必须来自搜索结果，不能编造。
4. 资料不足时明确说明。
5. 不能使用外部知识补充文档里没有的信息。
6. 回答结尾使用：

Sources:
- 文件名
"""
