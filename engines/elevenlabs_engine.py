import io
import uuid
import soundfile
import os
import time
from dotenv import load_dotenv
from typing import Dict, Any, Callable, Tuple
from elevenlabs import ElevenLabs

load_dotenv()


def create_app(**kwargs) -> Callable:
    model_path = kwargs.get("model_path", "scribe_v1")

    print('Initializing ElevenLabs client, model_path: ', model_path)

    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

    def transcribe(entry):
        if isinstance(entry, list):
            return [transcribe(e) for e in entry]

        audio_path = entry["audio"].get("path") or str(uuid.uuid4())
        stem = os.path.splitext(os.path.basename(audio_path))[0]

        wav_buffer = io.BytesIO()
        soundfile.write(wav_buffer, entry["audio"]["array"], entry["audio"]["sampling_rate"], format="WAV")
        wav_buffer.seek(0)

        try:
            start_time = time.time()
            response = client.speech_to_text.convert(
                model_id=model_path,
                file=(f"{stem}.wav", wav_buffer, "audio/wav"),
                language_code="heb",
                tag_audio_events=False,
            )
            transcription_time = time.time() - start_time
            return response.text, transcription_time
        except Exception as e:
            print(f"Exception calling ElevenLabs API: {e}")
            raise e

    return transcribe