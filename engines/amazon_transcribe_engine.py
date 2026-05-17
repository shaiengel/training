import boto3
import io
import os
import soundfile
import librosa
import time
import requests
import uuid
import asyncio
from typing import Dict, Any, Callable, Tuple
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

AWS_PROFILE = "portal"


def ensure_transcription_bucket(s3_client):
    from botocore.exceptions import ClientError

    bucket_name = "portal-evaluation-dataset"
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        return bucket_name
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("404", "NoSuchBucket"):
            raise

    region = s3_client.meta.region_name
    if region == "us-east-1":
        s3_client.create_bucket(Bucket=bucket_name)
    else:
        s3_client.create_bucket(
            Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": region}
        )
    lifecycle_policy = {
        "Rules": [{"ID": "ExpireObjectsAfter2Days", "Prefix": "", "Status": "Enabled", "Expiration": {"Days": 2}}]
    }
    s3_client.put_bucket_lifecycle_configuration(Bucket=bucket_name, LifecycleConfiguration=lifecycle_policy)
    return bucket_name


async def process_stream_audio(audio_bytes, region="us-east-1"):
    client = TranscribeStreamingClient(region=region)
    stream = await client.start_stream_transcription(
        language_code="he-IL",
        media_sample_rate_hz=16000,
        media_encoding="pcm",
    )

    transcript = []

    class EventHandler(TranscriptResultStreamHandler):
        async def handle_transcript_event(self, transcript_event: TranscriptEvent):
            results = transcript_event.transcript.results
            for result in results:
                if not result.is_partial:
                    for alt in result.alternatives:
                        transcript.append(alt.transcript)

    async def write_chunks():
        chunk_size = 1024 * 16
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i : i + chunk_size]
            await stream.input_stream.send_audio_event(audio_chunk=chunk)
        await stream.input_stream.end_stream()

    handler = EventHandler(stream.output_stream)
    await asyncio.gather(write_chunks(), handler.handle_events())
    return " ".join(transcript)


def create_app(**kwargs) -> Callable:
    model_type = kwargs.get("model_path")  # 'batch' or 'stream'
    session = boto3.Session(profile_name=AWS_PROFILE)
    transcribe_client = session.client("transcribe")

    if model_type == "batch":
        s3_client = session.client("s3")
        bucket_name = ensure_transcription_bucket(s3_client)

        def transcribe_batch(entry):
            if isinstance(entry, list):
                return [transcribe_batch(e) for e in entry]

            audio_path = entry["audio"].get("path") or str(uuid.uuid4())
            stem = os.path.splitext(os.path.basename(audio_path))[0]
            object_key = f"audio/{stem}.wav"

            audio_data = librosa.resample(
                entry["audio"]["array"], orig_sr=entry["audio"]["sampling_rate"], target_sr=16000
            )
            if len(audio_data) / 16000 < 0.5:
                return "", 0.0

            wav_buffer = io.BytesIO()
            soundfile.write(wav_buffer, audio_data, 16000, format="WAV")
            audio_bytes = wav_buffer.getvalue()

            s3_client.put_object(Bucket=bucket_name, Key=object_key, Body=audio_bytes)

            job_name = f"transcription-job-{stem}-{uuid.uuid4().hex[:8]}"
            transcribe_client.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={"MediaFileUri": f"s3://{bucket_name}/{object_key}"},
                MediaFormat="wav",
                LanguageCode="he-IL",
            )

            start_time = time.time()
            while True:
                status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
                if status["TranscriptionJob"]["TranscriptionJobStatus"] in ["COMPLETED", "FAILED"]:
                    break
                time.sleep(5)

            if status["TranscriptionJob"]["TranscriptionJobStatus"] == "COMPLETED":
                transcript_uri = status["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
                response = requests.get(transcript_uri)
                transcription_time = time.time() - start_time
                return response.json()["results"]["transcripts"][0]["transcript"], transcription_time
            else:
                raise Exception(f"Transcription job failed: {status}")

        return transcribe_batch

    elif model_type == "stream":

        def transcribe_stream(entry):
            if isinstance(entry, list):
                return [transcribe_stream(e) for e in entry]

            audio_data = librosa.resample(
                entry["audio"]["array"], orig_sr=entry["audio"]["sampling_rate"], target_sr=16000
            )
            if len(audio_data) / 16000 < 0.5:
                return "", 0.0

            wav_buffer = io.BytesIO()
            soundfile.write(wav_buffer, audio_data, 16000, format="WAV")
            audio_bytes = wav_buffer.getvalue()

            start_time = time.time()
            result = asyncio.run(process_stream_audio(audio_bytes))
            transcription_time = time.time() - start_time
            return result, transcription_time

        return transcribe_stream

    else:
        raise ValueError("model_type must be 'stream' or 'batch'")
