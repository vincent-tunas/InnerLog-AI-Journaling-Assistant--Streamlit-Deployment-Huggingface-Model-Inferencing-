"""SQLite storage is used to store the users Journal entry"""
"""SQLite is a lightweight database stored in user's localdisk"""


# 1. Import necessary modules
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3

# 2. set up the database path
def database_path():
    return Path(os.environ.get('INNERLOG_DB_PATH', Path(__file__).parent / 'data' / 'innerlog.db')).expanduser()


# 3. Define a context manager for database connections
@contextmanager
def connection():
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def initialize_database():
    with connection() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS journal_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL CHECK(length(trim(content)) > 0),
            created_at TEXT NOT NULL
        )''')
        # Remove the old field while preserving entries, IDs, and analysis links.
        columns = {row['name'] for row in conn.execute('PRAGMA table_info(journal_entries)')}
        if 'mood' in columns:
            conn.execute('ALTER TABLE journal_entries DROP COLUMN mood')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_entries_created ON journal_entries(created_at DESC, id DESC)')
        # Additive migration: existing journal entries are kept as they are.
        conn.execute('''CREATE TABLE IF NOT EXISTS entry_analysis (
            entry_id INTEGER PRIMARY KEY
                REFERENCES journal_entries(id) ON DELETE CASCADE,
            sentiment TEXT NOT NULL
                CHECK(sentiment IN ('positive','neutral','negative','mixed','unclear')),
            emotions_json TEXT NOT NULL,
            explanation TEXT NOT NULL,
            evidence_json TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            analyzed_at TEXT NOT NULL
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS entry_reflections (
            entry_id INTEGER PRIMARY KEY REFERENCES journal_entries(id) ON DELETE CASCADE,
            reflection TEXT NOT NULL,
            question TEXT NOT NULL,
            sources_json TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            generated_at TEXT NOT NULL
        )''')


def save_entry(title, content):
    content = content.strip()
    if not content:
        raise ValueError('Write something before saving your entry.')
    title = title.strip() or 'Untitled reflection'
    with connection() as conn:
        cursor = conn.execute(
            'INSERT INTO journal_entries (title, content, created_at) VALUES (?, ?, ?)',
            (title, content, datetime.now(timezone.utc).isoformat()),
        )
        return cursor.lastrowid


def list_entries(limit=None):
    with connection() as conn:
        sql = '''SELECT j.id, j.title, j.content, j.created_at,
                        a.sentiment, a.emotions_json, a.explanation,
                        a.evidence_json, a.model, a.prompt_version, a.analyzed_at,
                         r.reflection, r.question, r.sources_json,
                         r.model AS reflection_model, r.prompt_version AS reflection_prompt_version,
                         r.generated_at
                 FROM journal_entries AS j
                 LEFT JOIN entry_analysis AS a ON a.entry_id = j.id
                 LEFT JOIN entry_reflections AS r ON r.entry_id = j.id
                 ORDER BY j.created_at DESC, j.id DESC'''
        rows = conn.execute(sql if limit is None else sql + ' LIMIT ?', () if limit is None else (limit,)).fetchall()
        entries = []
        for row in rows:
            entry = {key: row[key] for key in ('id', 'title', 'content', 'created_at')}
            entry['analysis'] = None
            if row['sentiment'] is not None:
                entry['analysis'] = {
                    'sentiment': row['sentiment'],
                    'emotions': json.loads(row['emotions_json']),
                    'explanation': row['explanation'],
                    'evidence': json.loads(row['evidence_json']),
                    'model': row['model'],
                    'prompt_version': row['prompt_version'],
                    'analyzed_at': row['analyzed_at'],
                }
            entry['reflection'] = None
            if row['reflection'] is not None:
                entry['reflection'] = {
                    'reflection': row['reflection'], 'question': row['question'],
                    'sources': json.loads(row['sources_json']),
                    'model': row['reflection_model'],
                    'prompt_version': row['reflection_prompt_version'],
                    'generated_at': row['generated_at'],
                }
            entries.append(entry)
        return entries


def save_entry_analysis(entry_id, analysis):
    """Store one validated agent result per entry; re-analysis updates that row."""
    if not isinstance(analysis, dict):
        raise ValueError('Analysis must be a dictionary from result.to_dict().')
    if analysis.get('sentiment') not in ('positive', 'neutral', 'negative', 'mixed', 'unclear'):
        raise ValueError('Invalid sentiment label.')
    for key in ('explanation', 'model', 'prompt_version', 'analyzed_at'):
        if not isinstance(analysis.get(key), str) or not analysis[key].strip():
            raise ValueError(f'Analysis requires {key}.')
    for key in ('emotions', 'evidence'):
        items = analysis.get(key)
        if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
            raise ValueError(f'{key} must be a list of strings.')
    with connection() as conn:
        conn.execute('''
            INSERT INTO entry_analysis
                (entry_id, sentiment, emotions_json, explanation, evidence_json,
                 model, prompt_version, analyzed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(entry_id) DO UPDATE SET
                sentiment = excluded.sentiment,
                emotions_json = excluded.emotions_json,
                explanation = excluded.explanation,
                evidence_json = excluded.evidence_json,
                model = excluded.model,
                prompt_version = excluded.prompt_version,
                analyzed_at = excluded.analyzed_at
        ''', (
            entry_id, analysis['sentiment'],
            json.dumps(analysis['emotions'], ensure_ascii=False),
            analysis['explanation'],
            json.dumps(analysis['evidence'], ensure_ascii=False),
            analysis['model'], analysis['prompt_version'], analysis['analyzed_at'],
        ))


def save_entry_reflection(entry_id, result):
    """Save a successful response only; repeated calls overwrite the old reflection."""
    if not isinstance(result, dict):
        raise ValueError("Reflection must be a dictionary")
    for key in ('reflection', 'question', 'model', 'prompt_version', 'generated_at'):
        if not isinstance(result.get(key), str) or not result[key].strip():
            raise ValueError(f"Reflection requires {key}")
    sources = result.get('sources')
    if not isinstance(sources, list) or any(not isinstance(item, dict) for item in sources):
        raise ValueError('Reflection sources must be a list of dictionaries')
    with connection() as conn:
        conn.execute('''INSERT INTO entry_reflections
            (entry_id, reflection, question, sources_json, model, prompt_version, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(entry_id) DO UPDATE SET
                reflection=excluded.reflection, question=excluded.question,
                sources_json=excluded.sources_json, model=excluded.model,
                prompt_version=excluded.prompt_version, generated_at=excluded.generated_at
        ''', (entry_id, result['reflection'], result['question'],
              json.dumps(sources, ensure_ascii=False), result['model'],
              result['prompt_version'], result['generated_at']))
