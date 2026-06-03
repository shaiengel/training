import time
from typing import Callable

import librosa
import torch
from transformers import pipeline


def create_app(**kwargs) -> Callable:
    model_path = kwargs.get("model_path")
    device: str = kwargs.get("device", "auto")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        device=device,
        torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32,
        generate_kwargs={"language": "he", "num_beams": 5},
        chunk_length_s=30,
        stride_length_s=5,
    )

    def transcribe(entry):
        if isinstance(entry, list):
            return [transcribe(e) for e in entry]

        try:
            audio_resample = librosa.resample(
                entry["audio"]["array"], orig_sr=entry["audio"]["sampling_rate"], target_sr=16000
            )

            start_time = time.time()
            result = pipe({"array": audio_resample, "sampling_rate": 16000})
            transcription_time = time.time() - start_time

            return result["text"], transcription_time
        except Exception as e:
            print(f"Exception in transformers transcribe: {e}")
            raise e

    return transcribe
