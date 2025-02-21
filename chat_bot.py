import gradio as gr
import torch
import pyaudio
import wave
import numpy as np
import whisper
import time
import requests
import json
import re
from kokoro import generate_full
from models import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 500
PAUSE_DURATION = 2
SAMPLE_RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
OUTPUT_WAVE_FILENAME = "recorded_audio.wav"
OLLAMA_URL = "http://localhost:11434/api/generate"

KOKORO_MODEL = build_model("tts_model/kokoro-v0_19.pth", DEVICE)  
VOICEPACK = torch.load("voices/af.pt", weights_only=True).to(DEVICE)

def record_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=SAMPLE_RATE, input=True,
                        frames_per_buffer=CHUNK)
    
    frames = []
    silent_chunks = 0

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        audio_data = np.frombuffer(data, dtype=np.int16)
        if np.max(audio_data) < THRESHOLD:
            silent_chunks += 1
        else:
            silent_chunks = 0

        if silent_chunks > (SAMPLE_RATE / CHUNK * PAUSE_DURATION):
            break

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(OUTPUT_WAVE_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))

    return OUTPUT_WAVE_FILENAME

def transcribe_audio(audio_file):
    model = whisper.load_model("small.en")
    result = model.transcribe(audio_file)
    return result['text']

def query_ollama_streaming(prompt):
    headers = {'Content-Type': 'application/json'}
    data = {
        "prompt": prompt,
        "model": "mistral:latest"
    }

    response = requests.post("http://localhost:11434/api/generate", json=data, headers=headers, stream=True)

    print("Response Status Code:", response.status_code)

    if response.status_code == 200:
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    line_data = json.loads(line.decode('utf-8'))  # Decode each line separately
                    full_response += line_data.get("response", "")
                except json.JSONDecodeError:
                    print("Error decoding a streaming response line.")
        return full_response
    else:
        return f"Error: {response.status_code} - {response.text}"
def clean_text(text):

    return re.sub(r"[^a-zA-Z0-9\s.,'!()+=$@-]", "", text)

def text_to_speech(text):
    cleaned_text = clean_text(text)
    audio_data, _ = generate_full(KOKORO_MODEL, cleaned_text, VOICEPACK, lang='a', speed=1)
    audio_data = (audio_data / np.max(np.abs(audio_data)) * 32767).astype(np.int16)
    wav_filename = "output_tts.wav"
    with wave.open(wav_filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(audio_data.tobytes())
    return wav_filename

def process_text_input(text):
    response = query_ollama_streaming(text)
    audio_file = text_to_speech(response)
    return response, audio_file

def process_voice_input():
    audio_file = record_audio()
    transcription = transcribe_audio(audio_file)
    response, audio_output = process_text_input(transcription)
    return transcription, response, audio_output

with gr.Blocks() as demo:
    gr.Markdown("# SPECTRA MINDS CHATBOT")
    
    with gr.Tab("Text Input"):
        text_input = gr.Textbox(label="Enter your text")
        text_submit = gr.Button("Generate")
        text_output = gr.Textbox(label="Response")
        audio_output = gr.Audio(label="TTS Output", autoplay=True)
        text_submit.click(process_text_input, inputs=text_input, outputs=[text_output, audio_output])
    
    with gr.Tab("Voice Input"):
        record_button = gr.Button("Record & Process")
        transcribed_text = gr.Textbox(label="Transcription")
        voice_response = gr.Textbox(label="Response")
        voice_audio_output = gr.Audio(label="TTS Output", autoplay=True)
        record_button.click(process_voice_input, inputs=[], outputs=[transcribed_text, voice_response, voice_audio_output])
    
    gr.Markdown("Developed by SPECTRAMINDS")

demo.launch()
