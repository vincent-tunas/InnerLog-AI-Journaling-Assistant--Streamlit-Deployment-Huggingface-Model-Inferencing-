"""InnerLog RAG with Hugging Face sentence-transformers embeddings + ChromaDB.

Commands:
    python rag.py                   # status (does not load the embedding model)
    python rag.py --index           # explicitly index new PDF chunks
    python rag.py --test            # test search using the existing index
    python rag.py --test --query "How can I cope with stress?"

This is a LOCAL embedding model downloaded from Hugging Face on first use.
It does not send journal entries to Hugging Face Inference Providers.
The Reflection Agent can continue to import retrieve_sources(query, k=3).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _configured_path(name: str, default: Path) -> Path:
    value = Path(os.getenv(name, str(default))).expanduser()
    return value if value.is_absolute() else BASE_DIR / value


# A separate directory prevents accidental mixing with the existing Ollama index.
CHROMA_DIR = _configured_path("INNERLOG_CHROMA_DIR", BASE_DIR / "chroma_hf_db")
PDF_DIR = _configured_path("INNERLOG_KNOWLEDGE_DIR", BASE_DIR / "knowledge-base")
EMBEDDING_MODEL = os.getenv("INNERLOG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("INNERLOG_CHROMA_COLLECTION", "innerlog_hf")


class KnowledgeBaseError(RuntimeError):
    """RAG index, embedding, or knowledge-base operation failed."""


def _embeddings():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise KnowledgeBaseError(
            "Missing Hugging Face embedding dependency. Run: "
            "python -m pip install langchain-huggingface sentence-transformers"
        ) from exc
    try:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        raise KnowledgeBaseError(
            f"Could not load embedding model '{EMBEDDING_MODEL}': {exc}"
        ) from exc


def _store(create: bool = False):
    if not create and (not CHROMA_DIR.is_dir() or not any(CHROMA_DIR.iterdir())):
        raise KnowledgeBaseError(
            f"No ChromaDB index found at {CHROMA_DIR}. "
            "Run `python rag.py --index` once to build the Hugging Face index."
        )
    try:
        from langchain_chroma import Chroma
        return Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DIR),
            embedding_function=_embeddings(),
            collection_metadata={"hnsw:space": "cosine"},
        )
    except KnowledgeBaseError:
        raise
    except Exception as exc:
        raise KnowledgeBaseError(f"Cannot open ChromaDB at {CHROMA_DIR}: {exc}") from exc


def get_vectorstore():
    """Open the existing vector index without reindexing any PDF documents."""
    store = _store(create=False)
    try:
        count = store._collection.count()
    except Exception as exc:
        raise KnowledgeBaseError(f"Could not read collection '{COLLECTION_NAME}': {exc}") from exc
    if not count:
        raise KnowledgeBaseError(
            f"Collection '{COLLECTION_NAME}' contains no document chunks. "
            "Check your settings or run `python rag.py --index`."
        )
    return store


def retrieve_sources(query: str, k: int = 3) -> list[dict]:
    """Return the original Reflection Agent's expected source dictionary format."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Retrieval query must be non-empty.")
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be an integer greater than zero.")
    try:
        matches = get_vectorstore().similarity_search(query, k=k)
        return [
            {
                "text": doc.page_content,
                "source": str(doc.metadata.get("source") or "Unknown document"),
                "page": doc.metadata.get("page"),
            }
            for doc in matches
        ]
    except KnowledgeBaseError:
        raise
    except Exception as exc:
        raise KnowledgeBaseError(f"Knowledge retrieval failed: {exc}") from exc


def retrieve(query: str, k: int = 3) -> str:
    """Backward-compatible text-only retrieval."""
    return "\n\n".join(item["text"] for item in retrieve_sources(query, k=k))


def index_pdfs() -> int:
    """Explicitly index PDFs, skipping chunks already stored by this indexer."""
    if not PDF_DIR.is_dir():
        raise KnowledgeBaseError(f"PDF directory does not exist: {PDF_DIR}")
    paths = sorted(PDF_DIR.rglob("*.pdf"))
    if not paths:
        raise KnowledgeBaseError(f"No PDFs found in {PDF_DIR}")
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise KnowledgeBaseError(
            "Missing PDF indexing dependency. Install: "
            "python -m pip install langchain-community langchain-text-splitters pypdf"
        ) from exc

    store = _store(create=True)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    inserted = 0
    for path in paths:
        docs, ids = [], []
        try:
            for page in PyPDFLoader(str(path)).lazy_load():
                if not page.page_content.strip():
                    continue
                for chunk in splitter.split_documents([page]):
                    chunk.metadata["source"] = str(path.relative_to(PDF_DIR))
                    # Identifiers are deterministic, so repeated indexing skips same chunks.
                    digest = hashlib.sha256(
                        (str(path.resolve()) + ":" + str(chunk.metadata.get("page"))
                         + ":" + chunk.page_content).encode("utf-8")
                    ).hexdigest()
                    docs.append(chunk)
                    ids.append(digest)
            for start in range(0, len(ids), 100):
                chunk_ids = ids[start:start + 100]
                chunk_docs = docs[start:start + 100]
                existing = set(store.get(ids=chunk_ids, include=[])["ids"])
                missing = [(uid, doc) for uid, doc in zip(chunk_ids, chunk_docs) if uid not in existing]
                if missing:
                    store.add_documents(
                        [doc for _, doc in missing],
                        ids=[uid for uid, _ in missing],
                    )
                    inserted += len(missing)
            print(f"Processed {path.name}: {len(ids)} chunks")
        except Exception as exc:
            raise KnowledgeBaseError(f"Could not index '{path.name}': {exc}") from exc
    return inserted


def status():
    print("InnerLog RAG — Hugging Face local embeddings")
    print(f"PDF directory: {PDF_DIR}")
    print(f"Index directory: {CHROMA_DIR}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"PDF count: {len(list(PDF_DIR.rglob('*.pdf'))) if PDF_DIR.is_dir() else 0}")
    print(f"Index directory exists: {CHROMA_DIR.is_dir()}")
    print("Run `python rag.py --index` to create/update the index, or `python rag.py --test` to search.")


def main() -> int:
    parser = argparse.ArgumentParser(description="InnerLog Hugging Face embedding RAG")
    parser.add_argument("--index", action="store_true", help="Index only new PDF chunks")
    parser.add_argument("--test", action="store_true", help="Run a sample retrieval query")
    parser.add_argument("--query", default="How can I cope with stress and anxiety?", help="Test query")
    args = parser.parse_args()
    if args.index and args.test:
        parser.error("Use --index and --test in separate commands.")
    try:
        if args.index:
            print(f"New chunks indexed: {index_pdfs()}")
        elif args.test:
            matches = retrieve_sources(args.query, k=3)
            print(f"Found {len(matches)} source(s).")
            for n, item in enumerate(matches, 1):
                print(f"[{n}] {item['source']} page={item['page']}: {item['text'][:260]}")
        else:
            status()
    except KnowledgeBaseError as exc:
        print(f"Knowledge base error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
