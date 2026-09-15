"""Retrieval backend for mini_rag2.

这个文件负责 RAG 后端：
1. 把 Markdown 切成 parent / child。
2. 把 child 写入 Qdrant 向量库。
3. 把 parent 写入 JSON，供工具按 parent_id 取回完整上下文。
4. 暴露 LangChain tools 给 Agent 调用。
"""

import json
import shutil
import sys

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from mini_utils import EVIDENCE_SEPARATOR
from mini_config import (
    CHILD_OVERLAP,
    CHILD_SIZE,
    COLLECTION,
    DENSE_MODEL,
    DOCS_DIR,
    HEADERS_TO_SPLIT_ON,
    MIN_PARENT_SIZE,
    PARENT_SIZE,
    PARENT_STORE_PATH,
    QDRANT_PATH,
    QUERY_MODE,
    SPARSE_MODEL,
    SPARSE_VECTOR_NAME,
    TOP_K,
)


embeddings = HuggingFaceEmbeddings(model_name=DENSE_MODEL)
sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL)

vector_store = None
parent_store = None

def format_evidence_block(
    rank: int,
    evidence_type: str,
    source: str,
    parent_id: str,
    content: str,
    score=None,
    child_index=None,
) -> str:
    """把检索结果格式化成稳定证据块，方便 Agent 阅读和后续解析。"""

    lines = [
        f"### EVIDENCE {rank}",
        f"type: {evidence_type}",
        f"source: {source}",
        f"parent_id: {parent_id}",
    ]

    if child_index is not None:
        lines.append(f"child_index: {child_index}")

    if score is not None:
        lines.append(f"score: {score:.6f}")

    lines.extend(
        [
            "content:",
            content.strip(),
        ]
    )
    return "\n".join(lines)


def merge_metadata(first: dict, second: dict) -> dict:
    """合并相邻 parent 的标题 metadata，比如：上午 -> 中午。"""

    result = dict(first)

    for key, value in second.items():
        if key not in result:
            result[key] = value
            continue

        old_values = str(result[key]).split(" -> ")
        new_values = str(value).split(" -> ")
        result[key] = " -> ".join(
            dict.fromkeys(old_values + new_values)
        )

    return result


def merge_small_parent_docs(parent_docs):
    """顺序合并过短的 parent，减少碎片化上下文。"""

    merged = []
    current = None

    for doc in parent_docs:
        doc_copy = Document(
            page_content=doc.page_content,
            metadata=dict(doc.metadata),
        )

        if current is None:
            current = doc_copy
            continue

        combined_length = len(current.page_content) + 2 + len(doc_copy.page_content)

        if (
            len(current.page_content) < MIN_PARENT_SIZE
            and combined_length <= PARENT_SIZE
        ):
            current.page_content += "\n\n" + doc_copy.page_content
            current.metadata = merge_metadata(
                current.metadata,
                doc_copy.metadata,
            )
        else:
            merged.append(current)
            current = doc_copy

    if current is not None:
        can_merge_with_previous = (
            merged
            and len(current.page_content) < MIN_PARENT_SIZE
            and len(merged[-1].page_content) + 2 + len(current.page_content)
            <= PARENT_SIZE
        )

        if can_merge_with_previous:
            merged[-1].page_content += "\n\n" + current.page_content
            merged[-1].metadata = merge_metadata(
                merged[-1].metadata,
                current.metadata,
            )
        else:
            merged.append(current)

    return merged


def create_parent_child_chunks():
    """读取 Markdown，生成 parent_store 和 child_docs。"""

    md_files = sorted(DOCS_DIR.glob("*.md"))

    if not md_files:
        sys.exit(f"{DOCS_DIR} 中没有 Markdown 文件。")

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    large_parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_SIZE,
        chunk_overlap=0,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_SIZE,
        chunk_overlap=CHILD_OVERLAP,
    )

    parent_store_data = {}
    child_docs = []

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")

        section_docs = header_splitter.split_text(text)
        parent_docs = []

        for section_doc in section_docs:
            if len(section_doc.page_content) <= PARENT_SIZE:
                parent_docs.append(section_doc)
            else:
                parent_docs.extend(
                    large_parent_splitter.split_documents([section_doc])
                )

        print("\n[合并前 Parent]")
        for doc in parent_docs:
            print(
                f"标题={doc.metadata.get('H2', '无标题')}, "
                f"长度={len(doc.page_content)}"
            )

        parent_docs = merge_small_parent_docs(parent_docs)

        print("\n[合并后 Parent]")
        for doc in parent_docs:
            print(
                f"标题={doc.metadata.get('H2', '无标题')}, "
                f"长度={len(doc.page_content)}"
            )

        for parent_index, parent_doc in enumerate(parent_docs):
            parent_text = parent_doc.page_content
            parent_id = f"{md_file.stem}_p{parent_index}"

            parent_store_data[parent_id] = {
                "page_content": parent_text,
                "metadata": {
                    "source": md_file.name,
                    **parent_doc.metadata,
                },
            }

            child_texts = child_splitter.split_text(parent_text)

            for child_index, child_text in enumerate(child_texts):
                child_docs.append(
                    Document(
                        page_content=child_text,
                        metadata={
                            **parent_doc.metadata,
                            "source": md_file.name,
                            "parent_id": parent_id,
                            "child_index": child_index,
                        },
                    )
                )

    return parent_store_data, child_docs


def create_vector_store(client, retrieval_mode):
    """根据检索模式创建 QdrantVectorStore。"""

    if retrieval_mode == RetrievalMode.HYBRID:
        return QdrantVectorStore(
            client=client,
            collection_name=COLLECTION,
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_vector_name=SPARSE_VECTOR_NAME,
        )

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings,
        retrieval_mode=RetrievalMode.DENSE,
    )


def build_index(client: QdrantClient):
    """重建 parent JSON 和 child 向量索引。"""

    parent_store_data, child_docs = create_parent_child_chunks()

    PARENT_STORE_PATH.write_text(
        json.dumps(parent_store_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    vector_size = len(embeddings.embed_query("test"))

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=qmodels.VectorParams(
            size=vector_size,
            distance=qmodels.Distance.COSINE,
        ),
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: qmodels.SparseVectorParams(),
        },
    )

    store = create_vector_store(
        client,
        RetrievalMode.HYBRID,
    )
    store.add_documents(child_docs)

    print(
        f"[索引完成] "
        f"{len(parent_store_data)} 个 parent / "
        f"{len(child_docs)} 个 child"
    )

    for parent_id, parent in parent_store_data.items():
        child_count = sum(
            1
            for child in child_docs
            if child.metadata["parent_id"] == parent_id
        )
        print(
            f"  {parent_id}: "
            f"parent长度={len(parent['page_content'])}, "
            f"child数量={child_count}"
        )


def load_parent_store():
    """从 JSON 读取 parent_store。"""

    return json.loads(
        PARENT_STORE_PATH.read_text(encoding="utf-8")
    )


def retrieve_parents(good_child_hits, current_parent_store):
    """把检索到的 child 转换为对应的 parent，并去重。"""

    selected_parents = []
    seen_parent_ids = set()

    for child_doc, child_score in good_child_hits:
        parent_id = child_doc.metadata["parent_id"]

        if parent_id in seen_parent_ids:
            continue

        seen_parent_ids.add(parent_id)
        parent_data = current_parent_store.get(parent_id)

        if not parent_data:
            print(f"[警告] 找不到 parent：{parent_id}")
            continue

        selected_parents.append(
            {
                "parent_id": parent_id,
                "page_content": parent_data["page_content"],
                "source": parent_data["metadata"]["source"],
                "child_score": child_score,
            }
        )

    return selected_parents


@tool
def search_child_chunks(query:str,limit:int=TOP_K)->str:
    """在向量库里检索与 query 最相关的 child chunk，返回带 parent_id 的证据块。"""

    if vector_store is None:
        return "VECTOR_STORE_NOT_INITIALIZED"
    hits=vector_store.similarity_search_with_score(
        query,
        k=limit,
    )
    if not hits:
        return "NO_RELEVANT_CHUNKS"
    evidence_blocks=[]
    for rank,(doc,score) in enumerate(hits,start=1):
        evidence_blocks.append(
            format_evidence_block(
                rank=rank,
                evidence_type="child",
                source=doc.metadata.get("source",""),
                parent_id=doc.metadata.get("parent_id",""),
                child_index=doc.metadata.get("child_index",""),
                score=score,
                content=doc.page_content,
            )
        )
    return EVIDENCE_SEPARATOR.join(evidence_blocks)


@tool
def retrieve_parent_chunks(parent_id:str)->str:
    """按 parent_id 取回完整 parent 文档，用于补全 child 片段缺失的上下文。"""

    if parent_store is None:
        return "PARENT_STORE_NOT_INITIALIZED"
    parent_data=parent_store.get(parent_id)
    if not parent_data:
        return f"NO_PARENT_DOCUMENT:{parent_id}"
    return format_evidence_block(
        rank=1,
        evidence_type="parent",
        source=parent_data["metadata"].get("source",""),
        parent_id=parent_id,
        content=parent_data["page_content"],
    )


def setup_retrieval():
    """重建索引并返回可绑定给 LLM 的检索工具列表。"""

    global vector_store, parent_store

    # 学习阶段每次重建，避免旧数据干扰。
    # 必须先删除文件夹，再创建 QdrantClient。
    shutil.rmtree(QDRANT_PATH, ignore_errors=True)

    client = QdrantClient(path=str(QDRANT_PATH))
    build_index(client)

    vector_store = create_vector_store(
        client,
        QUERY_MODE,
    )
    parent_store = load_parent_store()

    return [
        search_child_chunks,
        retrieve_parent_chunks,
    ]
