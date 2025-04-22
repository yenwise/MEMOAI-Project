"""
server.py

This module implements the FastAPI server for MemoAI.
It defines API endpoints for audio transcription and text categorization,
serving as the backend interface for the front-end application.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from recorder import VoiceRecorder
from transcriptor import transcribe_audio
from category_classifier import categorize_text
from langchain_openai import ChatOpenAI
import shutil
import os
from pydantic import BaseModel
import json
from fastapi import FastAPI


app = FastAPI()

# Global variable (Recorder instance)
recorder = None
llm = ChatOpenAI(model="gpt-4", temperature=0)

@app.get("/")
async def root():
    return {"message": "MemoAI FastAPI Backend 🚀"}


@app.post("/record/start/")
async def start_recording():
    """
    Starts audio recording.
    """
    global recorder
    recorder = VoiceRecorder(output_filename="voice_prompt.wav")
    recorder.start_recording()
    return {"status": "recording", "message": "🔴 Recording started!"}


@app.post("/record/stop/")
async def stop_recording():
    """
    Stop audio recording.
    """
    global recorder
    if recorder:
        recorder.stop_recording()
        return {"status": "stopped", "message": "⏹️ Recording completed!", "file": "voice_prompt.wav"}
    else:
        return {"error": "Please start recording first!"}


@app.post("/transcribe/")
async def transcribe_audio_api(file: UploadFile = File(...)):
    try:
        """
        Converts the uploaded audio file to text.
        """
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Text
        transcribed_text = transcribe_audio(file_location)

        # Clean file
        os.remove(file_location)

        return {"text": transcribed_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

class CategorizeRequest(BaseModel):
    text: str

@app.post("/categorize/")
async def categorize_text_api(request: CategorizeRequest):
    category = categorize_text(request.text)
    return {"text": request.text, "category": category}

#@app.post("/categorize/")
#async def categorize_text_api(text: str):
   ## """
    #"""
    #category = categorize_text(text)
   # return {"text": text, "category": category}


@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    """
    An end-to-end API that handles recording an audio file, transcribing it into text,
    and categorizing the transcribed content using NLP.

    """
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 1️⃣ 
    transcribed_text = transcribe_audio(file_location)

    # 2️⃣ 
    category = categorize_text(transcribed_text)

    # 
    os.remove(file_location)

    return {
        "text": transcribed_text,
        "category": category
    }


conversation_memory = {}

class RefineRequest(BaseModel):
    session_id: str  
    text: str
    question: str = None

@app.post("/refine_note/")
async def refine_note(request: RefineRequest):
    """
    Allows the AI to understand previous context by storing the user's input.
    """
    if request.session_id not in conversation_memory:
        conversation_memory[request.session_id] = []

    # Allows the AI to understand the previous context by storing the user's input.
    conversation_memory[request.session_id].append(request.text)

    if request.question:
        prompt = f"""Based on the following previous notes and the user's new question, provide the best possible answer.

        **Previous Notes:** {conversation_memory[request.session_id]}
        **New Note:** {request.text}
        **User's Question:** {request.question}

        Answer:
        """
        response = llm.invoke(prompt)
        return {
        "text": request.text,
        "question": request.question,
        "response": response.content.strip()
        }

    else:
        follow_up_prompt = f"""
        **Previous Notes:** {conversation_memory[request.session_id]}
        **New Note:** {request.text}

        Analyze the previous answers. If the user has already provided information, **do not repeat the same question**.
        If information is missing, **ask only one meaningful follow-up question.**

        Respond in the following JSON format:
        {{
            "ai_response": "If there's no missing information, this will be the AI's answer.",
            "follow_up_question": "If there's missing info, write ONLY ONE follow-up question here."
        }}
        """
        ai_response = llm.invoke(follow_up_prompt).content.strip()

        return {
            "text": request.text,
            "ai_generated_question": ai_response
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
