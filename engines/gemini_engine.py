import io
import uuid
import soundfile
import os
import time
from dotenv import load_dotenv
from typing import Callable
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])


def create_app(**kwargs) -> Callable:
    model_path = kwargs.get("model_path", "gemini-2.0-flash")

    print("Initializing Gemini client, model_path: ", model_path)

    model = genai.GenerativeModel(model_path)

    def transcribe(entry):
        if isinstance(entry, list):
            return [transcribe(e) for e in entry]

        audio_path = entry["audio"].get("path") or str(uuid.uuid4())
        stem = os.path.splitext(os.path.basename(audio_path))[0]

        wav_buffer = io.BytesIO()
        soundfile.write(wav_buffer, entry["audio"]["array"], entry["audio"]["sampling_rate"], format="WAV")
        wav_buffer.seek(0)

        audio_part = {"mime_type": "audio/wav", "data": wav_buffer.read()}

        try:
            start_time = time.time()
            response = model.generate_content(
                [
                    audio_part,
                    "Transcribe this talmud hebrew audio exactly as spoken. Output only the transcription text, nothing else.",
                ]
            )
            transcription_time = time.time() - start_time
            return response.text.strip(), transcription_time
        except Exception as e:
            print(f"Exception calling Gemini API: {e}")
            raise e

    return transcribe
