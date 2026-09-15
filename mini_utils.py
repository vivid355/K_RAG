import json

import tiktoken
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

EVIDENCE_SEPARATOR = "\n\n---EVIDENCE---\n\n"
EVIDENCE_PREFIX = "### EVIDENCE"

PARENT_MESSAGE_NAMES = {"user", "assistant"}


def make_tool_call_key(tool_name: str, tool_args: dict) -> str:
    """Build a stable key used to avoid repeated tool calls."""

    if tool_name == "search_child_chunks":
        query = str(tool_args.get("query", " "))
        normalized_query = " ".join(query.lower().split())
        return f"search:{normalized_query}"

    if tool_name == "retrieve_parent_chunks":
        parent_id = str(tool_args.get("parent_id", "")).strip()
        return f"parent:{parent_id}"

    args_text = json.dumps(
        tool_args,
        ensure_ascii=False,
        sort_keys=True,
    )
    return f"{tool_name}:{args_text}"


def estimate_context_tokens(messages) -> int:
    """Estimate token count for a list of LangChain messages."""

    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return sum(
            len(encoding.encode(str(message.content)))
            for message in messages
            if getattr(message, "content", None)
        )
    except Exception:
        return sum(
            max(1, len(str(message.content)) // 2)
            for message in messages
            if getattr(message, "content", None)
        )


def extract_retrieval_contexts(messages) -> list[str]:
    """Collect useful retrieval evidence from ToolMessage values."""

    contexts = []
    ignored_prefixes = (
        "NO_RELEVANT_CHUNKS",
        "NO_PARENT_DOCUMENT",
        "VECTOR_STORE_NOT_INITIALIZED",
        "PARENT_STORE_NOT_INITIALIZED",
        "RETRIEVAL_ERROR:",
        "PARENT_RETRIEVAL_ERROR",
        "DUPLICATE_TOOL_CALL:",
        "UNKNOWN_TOOL:",
    )

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue

        content = str(message.content).strip()

        if not content:
            continue

        if content.startswith(ignored_prefixes):
            continue

        if EVIDENCE_SEPARATOR.strip() in content:
            parts = [
                part.strip()
                for part in content.split(EVIDENCE_SEPARATOR)
                if part.strip()
            ]
        elif content.startswith(EVIDENCE_PREFIX):
            parts = [content]
        else:
            parts = [content]

        contexts.extend(parts)

    return list(dict.fromkeys(contexts))

def is_plain_conversation_message(message) -> bool:
    """Return True for user/assistant chat messages, excluding tool chatter."""

    return (
        isinstance(message, (HumanMessage, AIMessage))
        and getattr(message, "name", None) in PARENT_MESSAGE_NAMES
        and not getattr(message, "tool_calls", None)
    )


def format_conversation(messages) -> str:
    """Format Human/AI messages as a compact transcript string."""

    lines = []

    for message in messages:
        if isinstance(message, HumanMessage):
            role = "User"
        elif isinstance(message, AIMessage):
            role = "Assistant"
        else:
            continue

        lines.append(f"{role}:{message.content}")

    return "\n".join(lines)


def remove_messages_not_in(messages, keep_ids):
    """Create RemoveMessage updates for messages not in keep_ids."""

    removals = []

    for message in messages:
        message_id = getattr(message, "id", None)

        if not message_id:
            continue

        if message_id not in keep_ids:
            removals.append(
                RemoveMessage(id=message_id)
            )

    return removals
