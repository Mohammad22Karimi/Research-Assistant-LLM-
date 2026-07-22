"""
Voice assistant utilities
=========================
Speech-to-text  : OpenAI Whisper (local, tiny model - fast on CPU)
Text-to-speech  : pyttsx3 (offline, no API key needed)

Why these choices?
    * Whisper "tiny" runs on any laptop CPU in a few seconds.
    * pyttsx3 uses the OS's built-in TTS engine - zero extra downloads.
    * Both are fully offline, matching the project's no-cloud constraint.
"""

import tempfile

import pyttsx3
import whisper

# Load Whisper once at import time (tiny = ~75 MB, fast on CPU).
_whisper_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        print("[voice] loading Whisper tiny model ...")
        _whisper_model = whisper.load_model("tiny")
    return _whisper_model


def transcribe(audio_path):
    """
    Purpose : Convert a recorded audio file to text using Whisper.
    Inputs  : audio_path (str) - path to a WAV/MP3/etc. audio file.
    Outputs : text (str) - the transcribed text.
    """
    result = _get_whisper().transcribe(audio_path)
    return result["text"].strip()


def speak(text):
    """
    Purpose : Read text aloud using the OS text-to-speech engine.
    Inputs  : text (str) - the text to speak.
    Outputs : none (audio plays through speakers).
    """
    engine = pyttsx3.init()
    engine.setProperty("rate", 180)   # words per minute
    engine.say(text)
    engine.runAndWait()
