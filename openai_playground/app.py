from typing import Iterator
import gradio as gr
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from openai._types import BinaryResponseContent
from pyaudio import PyAudio
from pydub import AudioSegment


load_dotenv()

client = OpenAI()
p = PyAudio()


def play_response(response: BinaryResponseContent):
    """
    Iterates through the response chunks, converts them to MP3 and plays them.
    """
    first_chunk = True

    # Stream and decode audio chunks as they are received
    for chunk in response.iter_bytes(chunk_size=1024*2**6):  # Adjust chunk size as needed
        if chunk:
            # Decode MP3 chunk to PCM using pydub
            audio = AudioSegment.from_file(BytesIO(chunk), format="mp3")

            if first_chunk:
                # Set up PyAudio with parameters from the first decoded audio chunk
                p = PyAudio()
                stream = p.open(format=p.get_format_from_width(audio.sample_width),
                                channels=audio.channels,
                                rate=audio.frame_rate,
                                output=True)
                first_chunk = False

            raw_data = audio.raw_data
            stream.write(raw_data)

    if not first_chunk:
        # Stop and close the stream
        stream.stop_stream()
        stream.close()

        # Terminate PyAudio
        p.terminate()

def text_to_speech(text: str) -> Iterator[bytes]:
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
    )

    yield from response.iter_bytes(chunk_size=1024*2**6)

def speech_to_text(filepath: str) -> str:
    with open(filepath, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )

        return transcript.text
    
def chat(message: str) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ]
    )

    return response.choices[0].message.content

def talk_to_ai(filepath: str) -> Iterator[bytes]:
    transcription = speech_to_text(filepath)
    response = chat(transcription)
    yield from text_to_speech(response)

demo = gr.Interface(
    fn=talk_to_ai, 
    inputs=gr.Microphone(type="filepath", format="mp3", label="Human input"), 
    outputs=gr.Audio(streaming=True, autoplay=True, label="AI output")
)

demo.launch(show_api=False, share=True)