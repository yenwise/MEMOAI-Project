"""
transcriptor.py

This module provides functionality to transcribe audio files using Whisper.
It exposes a function that takes an audio path and returns the transcribed text.
"""
import openai
import os
from dotenv import load_dotenv

# API Anahtarını Yükle
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def transcribe_audio(audio_file_path):
    """
    OpenAI Whisper API'nin yeni versiyonunu kullanarak sesi metne çevirir.
    """
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    with open(audio_file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"  # JSON yerine sadece text döndürmesi için
        )
    
    return response  # Yanıtın sadece metin olmasını sağladık

if __name__ == "__main__":
    text_result = transcribe_audio("voice_prompt.wav")
    print("📝 Parsed Text:", text_result)
