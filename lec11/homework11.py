import speech_recognition as sr
from pydub import AudioSegment
import os

def transcribe_wavefile(filename, language):
    """
    Transcribe an audio file (wav or m4a) into text.

    @params:
    filename (str) - the audio filename
    language (str) - language code, e.g., 'en-US', 'ja-JP'

    @returns:
    text (str) - recognized speech
    """
    # Convert to WAV if not already
    if not filename.lower().endswith(".wav"):
        wav_filename = "temp_audio.wav"
        audio = AudioSegment.from_file(filename)
        audio.export(wav_filename, format="wav")
        filename_to_use = wav_filename
    else:
        filename_to_use = filename

    recognizer = sr.Recognizer()

    # Load audio file
    with sr.AudioFile(filename_to_use) as source:
        audio_data = recognizer.record(source)

    # Try recognizing speech
    try:
        text = recognizer.recognize_google(audio_data, language=language)
    except sr.UnknownValueError:
        text = ""  # could not understand audio
    except sr.RequestError as e:
        text = f"API request failed: {e}"

    # Clean up temporary file
    if filename_to_use == "temp_audio.wav":
        os.remove(filename_to_use)

    return text

