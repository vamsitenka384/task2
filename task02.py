# Speech-to-Text Tool
# By CodTech Internship Project

# -------------------------
# Approach 1: Using SpeechRecognition (simple, requires internet for Google API)
# -------------------------

import speech_recognition as sr

def transcribe_with_speechrecognition(audio_file):
    """
    Transcribes short audio clips into text using SpeechRecognition + Google API.
    
    Args:
        audio_file (str): Path to the audio file (.wav, .aiff, .flac)

    Returns:
        str: Transcribed text
    """
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Speech not understood."
    except sr.RequestError:
        return "API unavailable."


# -------------------------
# Approach 2: Using Wav2Vec2 (offline, Hugging Face)
# -------------------------

from transformers import pipeline

# Load Wav2Vec2 model
asr_pipeline = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

def transcribe_with_wav2vec(audio_file):
    """
    Transcribes audio into text using Wav2Vec2 pretrained model.
    
    Args:
        audio_file (str): Path to the audio file (.wav required)

    Returns:
        str: Transcribed text
    """
    result = asr_pipeline(audio_file)
    return result['text']


# -------------------------
# Example Usage
# -------------------------
if
