import io
import uuid
import soundfile
import pydub
import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict, Any, Callable, Tuple

load_dotenv()


def create_app(**kwargs) -> Callable:
    model_path = kwargs.get("model_path", "whisper-1")  # Default to whisper-1 if not specified

    print('Initializing OpenAI client, model_path: ', model_path)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def transcribe(entry):
        if isinstance(entry, list):
            return [transcribe(e) for e in entry]

        audio_path = entry["audio"].get("path") or str(uuid.uuid4())
        stem = os.path.splitext(os.path.basename(audio_path))[0]

        # Convert audio to MP3 format
        wav_buffer = io.BytesIO()
        soundfile.write(wav_buffer, entry["audio"]["array"], entry["audio"]["sampling_rate"], format="WAV")
        wav_buffer.seek(0)

        # Convert WAV to MP3
        audio = pydub.AudioSegment.from_file(wav_buffer, format="wav")

        max_ms = 1390 * 1000
        chunks = [audio[i:i + max_ms] for i in range(0, len(audio), max_ms)]

        def transcribe_chunk(chunk, index):
            buf = io.BytesIO()
            chunk.export(buf, format="mp3")
            buf.seek(0)
            filename = f"{stem}_{index}.mp3" if len(chunks) > 1 else f"{stem}.mp3"
            return client.audio.transcriptions.create(
                model=model_path,
                file=(filename, buf, "audio/mpeg"),
                language="he",
            ).text

        try:
            start_time = time.time()
            parts = [transcribe_chunk(chunk, i) for i, chunk in enumerate(chunks)]
            transcription_time = time.time() - start_time
            return " ".join(parts), transcription_time
        except Exception as e:
            print(f"Exception calling OpenAI Whisper API: {e}")
            raise e

    return transcribe 