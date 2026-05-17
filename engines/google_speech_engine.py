import io
import soundfile
import uuid
import os
import librosa
import time
from typing import Dict, Any, Callable, Tuple
from google.cloud import speech, storage


def create_app(**kwargs) -> Callable:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
    if project is None:
        import google.auth
        creds, project = google.auth.default()
        if project is None:
            project = getattr(creds, "quota_project_id", None)
    speech_client = speech.SpeechClient()
    storage_client = storage.Client(project=project)
    bucket_name = f"{project}-stt-evaluation-audio"

    # Ensure bucket exists
    from google.api_core.exceptions import NotFound
    try:
        bucket = storage_client.get_bucket(bucket_name)
    except NotFound:
        bucket = storage_client.create_bucket(bucket_name)
        bucket.lifecycle_rules = [{"action": {"type": "Delete"}, "condition": {"age": 2}}]
        bucket.update()

    def transcribe(entry):
        if isinstance(entry, list):
            return [transcribe(e) for e in entry]

        audio_path = entry["audio"].get("path") or str(uuid.uuid4())
        stem = os.path.splitext(os.path.basename(audio_path))[0]
        blob_name = f"audio/{stem}.wav"
        blob = bucket.blob(blob_name)  
        if not blob.exists():          
            # Convert audio to proper format
            audio_data = librosa.resample(entry["audio"]["array"], orig_sr=entry["audio"]["sampling_rate"], target_sr=16000)

            # Save to temporary file
            tmp_path = os.path.join(os.environ.get("TEMP", "/tmp"), f"{stem}.wav")
            soundfile.write(tmp_path, audio_data, 16000, format="WAV")

            # Upload to GCS
            blob.upload_from_filename(tmp_path)
            os.remove(tmp_path)

        gcs_uri = f"gs://{bucket_name}/{blob_name}"
        audio = speech.RecognitionAudio(uri=gcs_uri)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="he-IL",
            model="default",
        )

        start_time = time.time()
        operation = speech_client.long_running_recognize(config=config, audio=audio)
        response = operation.result(timeout=5400)
        transcription_time = time.time() - start_time
        #blob.delete()

        return " ".join(result.alternatives[0].transcript for result in response.results), transcription_time

    return transcribe
