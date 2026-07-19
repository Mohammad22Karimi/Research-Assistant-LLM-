"""
Local LLM wrapper
=================
A very small wrapper around the local Llama 3.1 8B model served by Ollama.
Every part of the system that needs text generation goes through the single
function ask_llm(), so the model name and options live in exactly one place.

Requirements:
    1. Install Ollama from https://ollama.com
    2. Pull the model once:   ollama pull llama3.1:8b
    3. Ollama runs as a local server - no cloud, no API keys.
"""

import ollama

MODEL_NAME = "llama3.1:8b"


def ask_llm(prompt, system=None):
    """
    Purpose : Send a prompt to the local Llama 3.1 8B model and return its answer.
    Inputs  : prompt (str)  - the user/task prompt.
              system (str)  - optional system message that sets the model's role.
    Outputs : answer (str)  - the model's text response.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        options={"temperature": 0.2},  # low temperature -> factual, stable answers
    )
    return response["message"]["content"]
