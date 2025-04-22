"""
app.py

This module contains the main entry point of the Streamlit-based front-end
for MemoAI, an AI-powered voice note-taking assistant. It handles audio recording,
communication with the FastAPI backend, and rendering of categorized notes.
"""
import streamlit as st
import os
from recorder import VoiceRecorder
from transcriptor import transcribe_audio
from category_classifier import categorize_text

# Streamlit UI Title
st.title("🎤 MemoAI – AI-Powered Voice Note-Taking")

# Initialize recorder in Streamlit session state
if "recorder" not in st.session_state:
    st.session_state.recorder = None

# Start recording button
if st.button("🔴 Start Recording"):
    st.session_state.recorder = VoiceRecorder(output_filename="voice_prompt.wav")
    st.session_state.recorder.start_recording()
    st.warning("🎙️ Recording started...")

# Stop recording and process
if st.button("⏹️ Stop Recording and Process"):
    if st.session_state.recorder:
        st.session_state.recorder.stop_recording()
        st.success("✅ Recording completed!")

        # Transcribe audio file
        transcribed_text = transcribe_audio("voice_prompt.wav")
        st.subheader("📝 Transcribed Text:")
        st.write(transcribed_text)

        # Categorize the note
        category = categorize_text(transcribed_text)
        st.subheader("🏷️ Detected Category:")
        st.success(category)
    else:
        st.error("⚠️ Please start a recording first!")
