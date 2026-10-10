"""Local InnerLog knowledge retrieval.

Reading from Chroma never indexes PDFs. To index explicitly run `python rag.py --index`.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = Path(os.getenv('INNERLOG_CHROMA_DIR', str(BASE_DIR / 'chroma_db'))).expanduser()
PDF_DIR = Path(os.getenv('INNERLOG_KNOWLEDGE_DIR', str(BASE_DIR / 'knowledge-base'))).expanduser()
EMBEDDING_MODEL = os.getenv('INNERLOG_EMBEDDING_MODEL', 'nomic-embed-text')
COLLECTION_NAME = os.getenv('INNERLOG_CHROMA_COLLECTION', 'langchain')


class KnowledgeBaseError(RuntimeError):
    """Unable to access or search the local knowledge base."""


def get_vectorstore():
    """Open an existing persisted collection without indexing PDFs."""
    if not CHROMA_DIR.is_dir() or not any(CHROMA_DIR.iterdir()):
        raise KnowledgeBaseError(
            f'No ChromaDB index at {CHROMA_DIR}. Run `python rag.py --index` first, '
            'or set INNERLOG_CHROMA_DIR to your existing chroma_db folder.'
        )
    try:
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings
        store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DIR),
            embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
        )
        if store._collection.count() == 0:
            raise KnowledgeBaseError(
                'Chroma collection is empty. Verify INNERLOG_CHROMA_COLLECTION '
                '(default: langchain), or run `python rag.py --index`.'
            )
        return store
    except KnowledgeBaseError:
        raise
    except Exception as exc:
        raise KnowledgeBaseError(f'Unable to open ChromaDB: {exc}') from exc


def retrieve_sources(query: str, k: int = 3) -> list[dict]:
    """Return sources as dictionaries expected by ReflectionAgent.

    The query is embedded at search time; existing PDFs are not re-embedded.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError('Retrieval query must be non-empty.')
    if k < 1:
        raise ValueError('k must be at least 1.')
    try:
        docs = get_vectorstore().similarity_search(query, k=k)
        return [
            {
                'text': doc.page_content,
                'source': str(doc.metadata.get('source') or 'Unknown document'),
                'page': doc.metadata.get('page'),
            }
            for doc in docs
        ]
    except KnowledgeBaseError:
        raise
    except Exception as exc:
        raise KnowledgeBaseError(
            f'Knowledge search failed. Check Ollama is running and '
            f'`ollama pull {EMBEDDING_MODEL}` was completed: {exc}'
        ) from exc


def retrieve(query: str, k: int = 3) -> str:
    """Backward-compatible text-only retrieval."""
    return '\n\n'.join(source['text'] for source in retrieve_sources(query, k=k))


def index_pdfs() -> int:
    """Explicit, incremental indexing; never invoked on import or Streamlit reruns.

    Stable chunk IDs avoid re-inserting chunks already indexed with this function.
    Changed PDFs produce new IDs; remove stale entries separately if required.
    """
    if not PDF_DIR.is_dir():
        raise KnowledgeBaseError(f'PDF directory not found: {PDF_DIR}')
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings

    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    inserted = 0
    for path in sorted(PDF_DIR.rglob('*.pdf')):
        docs, ids = [], []
        for page in PyPDFLoader(str(path)).lazy_load():
            if not page.page_content.strip():
                continue
            for chunk in splitter.split_documents([page]):
                chunk.metadata['source'] = str(path.relative_to(PDF_DIR))
                identifier = hashlib.sha256(
                    (str(path.resolve()) + ':' + str(chunk.metadata.get('page'))
                     + ':' + chunk.page_content).encode('utf-8')
                ).hexdigest()
                docs.append(chunk)
                ids.append(identifier)
        if not ids:
            continue
        # Check IDs before embedding, so unchanged documents are never re-embedded.
        for start in range(0, len(ids), 100):
            batch_ids = ids[start:start + 100]
            batch_docs = docs[start:start + 100]
            existing = set(store.get(ids=batch_ids, include=[])['ids'])
            missing = [(identifier, doc) for identifier, doc in zip(batch_ids, batch_docs)
                       if identifier not in existing]
            if missing:
                store.add_documents(
                    [doc for _, doc in missing],
                    ids=[identifier for identifier, _ in missing],
                )
                inserted += len(missing)
        print(f'Indexed: {path.name}')
    return inserted


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='InnerLog knowledge base indexer')
    parser.add_argument('--index', action='store_true', help='Index new PDF chunks explicitly')
    args = parser.parse_args()
    if args.index:
        try:
            print(f'New chunks indexed: {index_pdfs()}')
        except KnowledgeBaseError as exc:
            parser.exit(1, f'Knowledge base error: {exc}\n')
    else:
        parser.print_help()
