from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    """Structured result returned by the query rewrite LLM call."""

    is_clear: bool = Field(
        description="用户问题是否清晰，是否可以直接用于文档检索"
    )
    questions: list[str] = Field(
        default_factory=list,
        description="改写后的、适合独立检索的问题列表，最多 3 个",
    )
    clarification_needed: str = Field(
        default="",
        description="如果问题不清晰，这里写需要用户补充什么信息",
    )
