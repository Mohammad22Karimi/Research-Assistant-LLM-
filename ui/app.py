"""
Browser-based interface for the Research Assistant.
Supports both typed questions and microphone (voice) input.

Run with:
    streamlit run ui/app.py

Streamlit then opens the app in your browser automatically
(usually http://localhost:8501).

"""

import os
import sys
import tempfile

# Allow "from rag import ..." etc. by adding the project root to sys.path,
# because Streamlit runs this file from inside the ui/ folder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from functions.comparison import compare_results
from functions.limitations import compare_limitations
from functions.references import compare_references
from functions.research_gap import find_research_gap
from functions.results_extraction import extract_experimental_results
from functions.survey import generate_survey
from rag import pipeline, router
from utils import voice


# ── Backend helpers ─────────────────────────────────────────────────────────

def dispatch(action, question):
    """
    Purpose : Map a router keyword to the matching function and return text.
    Inputs  : action (str)   - keyword chosen by the LLM router.
              question (str) - the original user question.
    Outputs : answer (str)   - text to display.
    """
    if action == "EXTRACT_RESULTS":
        df = extract_experimental_results()
        return df.to_string() if not df.empty else "No metrics found."
    elif action == "COMPARE_RESULTS":
        df = compare_results()
        text = df.to_string() if not df.empty else "No metrics found."
        return text + "\n\nChart saved to: comparison_chart.png"
    elif action == "GENERATE_SURVEY":
        return generate_survey()
    elif action == "RESEARCH_GAP":
        return find_research_gap()
    elif action == "COMPARE_REFERENCES":
        df = compare_references()
        return df.to_string() if not df.empty else "No references sections found."
    elif action == "COMPARE_LIMITATIONS":
        return compare_limitations()
    else:
        return pipeline.ask(question)


def answer_question(question):
    """
    Purpose : Route a question through the LLM router, then run it.
    Inputs  : question (str).
    Outputs : (action, answer) tuple of strings.
    """
    action = router.route(question)
    return action, dispatch(action, question)


# ── Page layout ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Research Assistant", page_icon="📚")
st.title("📚 LLM Research Assistant")
st.caption("Ask questions about your papers by typing or speaking.")

# Three tabs, mirroring the CLI: text Q&A, voice Q&A, and one-click actions.
tab_text, tab_voice, tab_actions = st.tabs(["💬 Text", "🎙️ Voice", "⚙️ Actions"])


# ── Tab 1: Text input ───────────────────────────────────────────────────────
with tab_text:
    question = st.text_area(
        "Your question",
        placeholder="e.g. Which paper has the best F1 score?",
        height=100,
    )
    if st.button("Ask", type="primary", key="ask_text"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking ..."):
                action, answer = answer_question(question)
            st.info(f"Router chose action: {action}")
            st.text_area("Answer", value=answer, height=300, key="ans_text")


# ── Tab 2: Voice input ──────────────────────────────────────────────────────
with tab_voice:
    st.write("Record your question. It will be transcribed, answered, and read aloud.")
    # st.audio_input records from the microphone and returns an audio file.
    audio = st.audio_input("Record your question")
    if st.button("Transcribe & Ask", type="primary", key="ask_voice"):
        if audio is None:
            st.warning("Please record something first.")
        else:
            # Whisper needs a real file path, so save the recording to a
            # temporary .wav file first.
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio.getvalue())
                tmp_path = tmp.name

            with st.spinner("Transcribing ..."):
                text = voice.transcribe(tmp_path)
            os.remove(tmp_path)
            st.text_input("Transcribed question", value=text, disabled=True)

            with st.spinner("Thinking ..."):
                action, answer = answer_question(text)
            st.info(f"Router chose action: {action}")
            st.text_area("Answer", value=answer, height=300, key="ans_voice")

            # Read the answer aloud (first 500 chars to keep it short).
            voice.speak(answer[:500])


# ── Tab 3: Quick actions (no question needed) ───────────────────────────────
with tab_actions:
    st.write("Run analysis functions directly without typing a question.")

    # Each button maps to one function. We store the result in session_state
    # so it stays visible after the rerun that a button click triggers.
    col1, col2, col3 = st.columns(3)
    if col1.button("Extract Results"):
        df = extract_experimental_results()
        st.session_state["action_out"] = df.to_string() if not df.empty else "No metrics found."
    if col2.button("Compare Results"):
        df = compare_results()
        text = df.to_string() if not df.empty else "No metrics found."
        st.session_state["action_out"] = text + "\n(Chart saved to comparison_chart.png)"
    if col3.button("Generate Survey"):
        st.session_state["action_out"] = generate_survey()

    col4, col5, col6 = st.columns(3)
    if col4.button("Research Gap"):
        st.session_state["action_out"] = find_research_gap()
    if col5.button("Compare References"):
        df = compare_references()
        st.session_state["action_out"] = df.to_string() if not df.empty else "No references found."
    if col6.button("Compare Limitations"):
        st.session_state["action_out"] = compare_limitations()

    st.text_area("Output", value=st.session_state.get("action_out", ""), height=400)

    # If a chart was produced by an action, show it below the text output.
    if os.path.exists("comparison_chart.png"):
        st.image("comparison_chart.png", caption="comparison_chart.png")
    if os.path.exists("references_chart.png"):
        st.image("references_chart.png", caption="references_chart.png")
