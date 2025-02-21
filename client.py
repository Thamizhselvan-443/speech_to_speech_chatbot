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

    print("\nListening... Speak now.")
    time.sleep(0.75) 

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
            print("Finished recording.")
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
    print("Transcribing audio...")
    model = whisper.load_model("small.en")
    result = model.transcribe(audio_file)
    return result['text']

def query_ollama_streaming(prompt):
    print("Querying Ollama model (streaming)...")
    headers = {'Content-Type': 'application/json'}
    data = {
        "prompt": f"***Generate 50 words only , strictly follow this ***, don't use this phrase {prompt}",
        "model": "mistral:latest"
    }
    response = requests.post(OLLAMA_URL, json=data, headers=headers, stream=True)

    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                line_data = json.loads(line.decode('utf-8'))
                yield line_data.get("response", "")
    else:
        yield f"Error: {response.status_code} - {response.text}"

def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9\s.,'!()+=$@-]", "", text)

def speak_paragraphs(paragraph_accumulator, model, voicepack):
    for paragraph in paragraph_accumulator:
        try:
            cleaned_text = clean_text(paragraph.strip())
            audio_data, _ = generate_full(model, cleaned_text, voicepack, lang='a', speed=1)
            audio_data = (audio_data / np.max(np.abs(audio_data)) * 32767).astype(np.int16)
            audio_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1,
                                                  rate=24000, output=True)
            audio_stream.write(audio_data.tobytes())
            audio_stream.close()
        except Exception as e:
            print(f"\nError in TTS: {e}")

def process_ollama_response(response_stream, model, voicepack):
    paragraph_accumulator = [] 
    current_paragraph = ""
    for response_part in response_stream:
        print(response_part, end="", flush=True)  
        current_paragraph += response_part
        if "\n\n\n" in current_paragraph:
            paragraph, _, current_paragraph = current_paragraph.partition("\n\n")
            paragraph_accumulator.append(paragraph.strip())
    if current_paragraph.strip():
        paragraph_accumulator.append(current_paragraph.strip())
    speak_paragraphs(paragraph_accumulator, model, voicepack)

def main():
    while True:
        print("""
        Enter a choice :
        1 - Text
        2 - Voice
        """)

        user_input = int(input("Enter a value : "))
        if user_input==2:
            audio_file = record_audio()
            transcription = transcribe_audio(audio_file)
            print(f"\nTranscription: {transcription}")
            if transcription.strip():
                print("Streaming response from Ollama...\n")
                response_stream = query_ollama_streaming(transcription)
                process_ollama_response(response_stream, KOKORO_MODEL, VOICEPACK)
            else:
                print("No speech detected.")
            user_input = input("\nType 'exit' to stop or press Enter to continue: ").strip().lower()
            if user_input == 'exit':
                print("Exiting...")
                break
        elif user_input==1:
            prompt = input("Enter your prompt : ")
            if prompt.strip():
                print("Streaming response from Ollama...\n")
                response_stream = query_ollama_streaming(prompt)
                process_ollama_response(response_stream, KOKORO_MODEL, VOICEPACK)
            else:
                print("No speech detected.")
            user_input = input("\nType 'exit' to stop or press Enter to continue: ").strip().lower()
            if user_input == 'exit':
                print("Exiting...")
                break


if __name__ == "__main__":
    main()
