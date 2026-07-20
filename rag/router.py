"""
Stage 6 - Function calling ROUTER
=================================
The project requires that "the language model automatically decides when a
function must be called".  We implement this with the simplest possible
mechanism - no agent framework:

    1. We show the LLM the user's question and a menu of available actions.
    2. The LLM must reply with EXACTLY ONE keyword.
    3. Plain Python then dispatches to the matching function.

This IS function calling: the model chooses the tool, the program executes
it.  It is just implemented transparently instead of hidden inside a
framework, which makes it easy to explain and to debug.
"""

from utils import llm

# The routing keywords and what they mean.  main.py maps each keyword to a
# real Python function.
ACTIONS = {
    "ANSWER": "Answer a normal question about the papers' content (default).",
    "EXTRACT_RESULTS": "Extract experimental metrics (accuracy, F1, mAP, ...) from the papers.",
    "COMPARE_RESULTS": "Compare the papers' performance with a table and a chart.",
    "GENERATE_SURVEY": "Write a mini survey of all papers.",
    "RESEARCH_GAP": "Find limitations, unsolved problems and future work.",
    "COMPARE_REFERENCES": "Compare the papers' reference lists by publication year.",
    "COMPARE_LIMITATIONS": "Compare the weaknesses/limitations of the methods.",
}

ROUTER_SYSTEM = (
    "You are a router. Read the user's request and decide which ONE action "
    "fits best. Reply with EXACTLY one keyword from the list and nothing else."
)


def build_router_prompt(question):
    """
    Purpose : Build the routing prompt shown to the LLM.
    Inputs  : question (str) - the user's request.
    Outputs : prompt (str).
    """
    menu = "\n".join(f"- {key}: {desc}" for key, desc in ACTIONS.items())
    return (
        f"Available actions:\n{menu}\n\n"
        f"User request: \"{question}\"\n\n"
        "Which action should be executed? Answer with one keyword only."
    )


def route(question):
    """
    Purpose : Let the LLM pick the action for a user request.
    Inputs  : question (str) - the user's request.
    Outputs : keyword (str) - one of the keys of ACTIONS.
              If the model answers something unexpected, we fall back to
              "ANSWER" so the system never crashes.
    """
    reply = llm.ask_llm(build_router_prompt(question), system=ROUTER_SYSTEM)
    reply = reply.strip().upper()

    # The model sometimes adds punctuation or extra words - accept the
    # keyword if it appears anywhere in the reply (longest keywords first,
    # so "COMPARE_RESULTS" is not mistaken for "ANSWER" etc.).
    for keyword in sorted(ACTIONS, key=len, reverse=True):
        if keyword in reply:
            return keyword
    return "ANSWER"  # safe default
