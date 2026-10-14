# InnerLog — repaired integration

This folder contains the five compatible Python modules. RAG queries your **existing**
Chroma collection (`langchain` by default) and does not re-embed PDFs when the app loads.

## Installation on Windows PowerShell

1. Back up the original folder, especially `data/innerlog.db` and `chroma_db/`.
2. Copy these five `.py` files into your existing `innerlog_reflection` project folder.
   Keep your original `data/`, `chroma_db/`, and `knowledge-base/` folders unchanged.
3. From the project directory, activate your existing virtual environment, then run:

   ```powershell
   python -m pip install -r requirements.txt
   ollama pull nomic-embed-text
   ollama pull gemma3:4b
   ollama list
   streamlit run app.py
   ```

Ollama must be running locally. Default sentiment and reflection models are both
`gemma3:4b`, while embeddings use `nomic-embed-text`.

## Existing index location

By default, RAG searches `<project folder>/chroma_db`, using collection `langchain`.
This matches the Chroma defaults used in the supplied RAG module. If the old index
lives elsewhere, set `INNERLOG_CHROMA_DIR` to its existing path.
If you used a different collection name set `INNERLOG_CHROMA_COLLECTION` as well.
**Do not delete your existing ChromaDB and do not re-index it unless required.**
If the DB was built using a different embedding model, set `INNERLOG_EMBEDDING_MODEL`
to the exact model originally used; otherwise similarity retrieval can fail.

## Index PDF files (only if needed)

If you do not have a persisted index yet, place PDFs inside `knowledge-base/` and run:

```powershell
python rag.py --index
```

This is a separate explicit operation, never triggered by starting the app.
On subsequent executions of this indexer, unchanged chunk IDs are skipped.
The supplied original indexer sometimes stored only the last page of each PDF;
if your old Chroma collection was built that way, it may have incomplete coverage.
You may explicitly run the indexer to add missing content, but old entries
using legacy random IDs may coexist with the new stable IDs. A clean rebuild
should only be done after backing up your current index.

## Quick sanity check

```powershell
python -c "from rag import retrieve_sources, KnowledgeBaseError; from reflection_agent import ReflectionAgent; print('Imports OK')"
python -c "from rag import retrieve_sources; print(retrieve_sources('coping with anxiety'))"
```

Saved reflections are generated only when the user presses **Generate reflection**.
Sentiment analysis is run after a journal entry is saved. The local reflection
agent uses sentiment, summarized prior analyses, and knowledge-base excerpts.

Do not treat the app's responses as diagnosis, treatment, or crisis support.
