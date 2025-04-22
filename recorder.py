"""
recorder.py

This module handles audio recording functionality for MemoAI.
It provides a VoiceRecorder class that can record microphone input 
and save it as a .wav file using PyAudio.
"""
import pyaudio
import wave
import time
import threading

class VoiceRecorder:
    def __init__(self, output_filename="voice_prompt.wav", rate=44100, chunk=1024, record_time=5):
        self.output_filename = output_filename
        self.rate = rate
        self.chunk = chunk
        self.frames = []
        self.recording = threading.Event()
        self.record_time = record_time

    def start_recording(self):
        self.recording.set()
        self.frames = []
        print(f"🔴 Recording started ({self.record_time} second)...")

        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=self.rate, input=True, frames_per_buffer=self.chunk)

        start_time = time.time()
        while self.recording.is_set():
            self.frames.append(stream.read(self.chunk, exception_on_overflow=False))
            if time.time() - start_time >= self.record_time:
                break

        stream.stop_stream()
        stream.close()
        audio.terminate()

        with wave.open(self.output_filename, "wb") as sound_file:
            sound_file.setnchannels(1)
            sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            sound_file.setframerate(self.rate)
            sound_file.writeframes(b''.join(self.frames))

        print(f"✅ Recording finished successfully: {self.output_filename}")

    def stop_recording(self):
        self.recording.clear()

if __name__ == "__main__":
    recorder = VoiceRecorder(record_time=5)
    recorder.start_recording()
