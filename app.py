'''Innerlog UI'''

# Import Modules
import streamlit as st
from datetime import datetime

import sqlite3
from database import (
    initialize_database, list_entries, save_entry, save_entry_analysis, save_entry_reflection,
)
from sentiment_agent import SentimentAgent, SentimentAgentError
from reflection_agent import ReflectionAgent, ReflectionAgentError


# 1. Initialize the database
try:
    initialize_database()
except (sqlite3.Error, OSError) as e:
    st.error(f"Error initializing the Journal database: {e}")
    st.stop()


# 2. Initialize the initial state of the journal entry form
if st.session_state.pop("entry_saved", False):
    st.session_state["journal_title"] = ""
    st.session_state["journal_draft"] = ""
    st.success("Your entry has been saved.")

analysis_notice = st.session_state.pop("analysis_notice", None)
if analysis_notice:
    level, message = analysis_notice
    if level == "success":
        st.success(message)
    else:
        st.warning(message)

# Keep form state when its widgets are hidden on the results page.
for key, default in (
    ("journal_title", ""), ("journal_draft", "")
):
    st.session_state[key] = st.session_state.get(key, default)

# 3. Define the function to analyze a saved journal entry

def analyze_saved_entry(entry_id, content):
    """Journal storage is already committed before this function runs."""
    try:
        with st.spinner("Analyzing your entry locally..."):
            result = SentimentAgent().analyze(content)
    except (ValueError, SentimentAgentError) as error:
        return "warning", f"Your entry is saved. Analysis could not finish: {error}"
    try:
        save_entry_analysis(entry_id, result.to_dict())
    except (sqlite3.Error, OSError, ValueError):
        return "warning", "Your entry is saved, but its analysis could not be stored. Please retry."
    return "success", "Sentiment analysis saved."

def reflect_on_entry(entry):
    """Generate on demand, not on each Streamlit rerun."""
    if not entry.get("analysis"):
        return "warning", "Run sentiment analysis for this entry first."
    try:
        with st.spinner("Retrieving local knowledge and preparing your reflection..."):
            result = ReflectionAgent().reflect(entry, history=load_entries(limit=10))
            save_entry_reflection(entry["id"], result.to_dict())
    except (ReflectionAgentError, ValueError, sqlite3.Error, OSError) as error:
        return "warning", f"Reflection could not be generated: {error}"
    return "success", "Your reflection has been saved."


def display_reflection(entry):
    reflection = entry.get("reflection")
    if reflection:
        st.markdown("**Personal reflection**")
        st.write(reflection["reflection"])
        st.markdown("**A question to consider**")
        st.write(reflection["question"])
        with st.expander("Knowledge base references"):
            for source in reflection["sources"]:
                page = source.get("page")
                page_label = f" (page {page + 1})" if isinstance(page, int) else ""
                st.caption(f"{source['source']}{page_label}")
            st.caption("References are retrieved passages, not verified citations for every claim.")
    else:
        st.caption("No reflection generated for this entry yet.")
    if entry.get("analysis") and st.button(
        "Regenerate reflection" if reflection else "Generate reflection",
        key=f"reflection_{entry['id']}",
    ):
        st.session_state["analysis_notice"] = reflect_on_entry(entry)
        st.rerun()


# 4. Define the function to load journal entries from the SQL database
def load_entries(limit=None):
    try:
        journal_entries = list_entries(limit=limit)
        return journal_entries
    except (sqlite3.Error, OSError):
        st.error("Could not load your journal entries. Please try again.")
        st.stop()


# 5. Display the 3 latest journal entries in the journal
def display_recent_journal_entries(full=False):
    st.subheader("Past Entries")
    displayed_entries = load_entries(limit=3 if not full else None)
    if not displayed_entries:
        st.info("No Journal Entries found. Write your first entry")
        return
    for entry in displayed_entries:
        with st.container(border=True):
            date = datetime.fromisoformat(entry["created_at"]).astimezone()
            st.caption(date.strftime("%d %b %Y · %H:%M"))
            st.text(entry["title"])
            
            if full:
                st.text(entry["content"])
            else:
                preview = entry["content"]
                st.text(preview[:200] + ("…" if len(preview) > 200 else ""))
                with st.expander("Read full entry"):
                    st.text(entry["content"])
            st.caption(f"{len(entry['content'].split())} words")
            display_reflection(entry)

"""Sentiment results page, rendered by app.py's sidebar toggle."""

# 6. Display the sentiment analysis results of journal entries

def render_sentiment_analysis_results_page(entries, analyze_saved_entry):
    st.title("Sentiment Analysis Results of Journal Entries")
    st.caption("Review the saved analysis of your journal entries.")

    if not entries:
        st.info("No journal entries yet. Switch to the journal page to write your first entry.")
        return

    for entry in entries:
        with st.container(border=True):
            st.text(entry["title"])
            created_at = datetime.fromisoformat(entry["created_at"]).astimezone()
            st.caption(created_at.strftime("%d %b %Y · %H:%M"))

            analysis = entry["analysis"]
            if analysis:
                st.subheader(f"Sentiment: {analysis['sentiment'].capitalize()}")
                st.write("Emotions: " + (
                    ", ".join(analysis["emotions"]) or "No specific emotions identified"
                ))
                st.text(analysis["explanation"])
                with st.expander("Supporting excerpts"):
                    for excerpt in analysis["evidence"]:
                        st.text(f'“{excerpt}”')
                analyzed_at = datetime.fromisoformat(analysis["analyzed_at"]).astimezone()
                st.caption("Analyzed: " + analyzed_at.strftime("%d %b %Y · %H:%M"))
            else:
                st.info("This entry has no saved sentiment analysis yet.")
                if st.button("Analyze entry", key=f"analyze_entry_{entry['id']}"):
                    st.session_state["analysis_notice"] = analyze_saved_entry(
                        entry["id"], entry["content"]
                    )
                    st.rerun()

            display_reflection(entry)

            with st.expander("Read journal entry"):
                st.text(entry["content"])


with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>InnerLog: AI Journaling Assistant</h1>", unsafe_allow_html=True)
    st.divider()
    show_sentiment_results = st.toggle(
        "Show sentiment analysis results",
        key="show_sentiment_results",
        help="On: sentiment results. Off: journal entry page.",
    )
    st.caption("Reflect on your journey")
    st.caption("Sentiment analysis and optional RAG reflections run locally through Ollama. They do not diagnose mental health.")
    st.divider()


if show_sentiment_results:
    render_sentiment_analysis_results_page(load_entries(), analyze_saved_entry)
else:
    st.subheader("Write your journal entry")
    with st.form("journal_form"):
        title = st.text_input("Title", key="journal_title")
        content = st.text_area("Content", key="journal_draft", height=200)
        submitted = st.form_submit_button("Save Entry")
        if submitted:
            if not title.strip() or not content.strip():
                st.warning("Please provide both a title and content for your journal entry.")
            else:
                try:
                    entry_id = save_entry(title, content)
                except (sqlite3.Error, OSError, ValueError) as e:
                    st.error(f"Error saving your journal entry: {e}")
                else:
                    st.session_state["analysis_notice"] = analyze_saved_entry(
                        entry_id, content.strip()
                    )
                    # Reset widget values at the top of the NEXT run, before creation.
                    st.session_state["entry_saved"] = True
                    st.rerun()
    display_recent_journal_entries()
