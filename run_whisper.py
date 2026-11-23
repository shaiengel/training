#!/usr/bin/env python3

import argparse
import logging

import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import pipeline
import faster_whisper

import torch

import io
import datasets
import jiwer
import json

import soundfile
import numpy
import pydub

import whisper
import whisper.normalizers

# Supported engines and models:
#
# 1. Engine: openai-whisper
#    - Models: large-v2, large-v3
# 2. Engine: transformers
#    - Models: openai/whisper-large-v2, openai/whisper-large-v3, user-trained models
# 3. Engine: faster-whisper
#    - Models: large-v2, large-v3, user-trained models


def initialize_model(
    engine,
    model_path,
    tuned_model_path,
    compute_type="float16",
    max_previous_tokens=128,
    fix_seek=True,
    log_windows=False,
):
    if engine == "openai-whisper":
        model = whisper.load_model(model_path)

        def transcribe(filename):
            return transcribe_openai_whisper(model, filename)

        return transcribe

    if engine == "openai-whisper-tuned":
        model = whisper.load_model(model_path)

        if tuned_model_path:
            tuned_model = WhisperForConditionalGeneration.from_pretrained(tuned_model_path)
            model = copy_hf_model_weights_to_openai_model_weights(tuned_model, model)

        def transcribe(filename):
            return transcribe_openai_whisper(model, filename)

        return transcribe

    if engine == "transformers":
        device = "cuda:0"
        torch_dtype = torch.float16

        model = WhisperForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, use_safetensors=True
        )
        model.to(device)
        processor = WhisperProcessor.from_pretrained(model_path)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=10,
            stride_length_s=(4, 2),
            batch_size=1,
            torch_dtype=torch_dtype,
            device=device,
        )

        def transcribe(filename):
            result = pipe(filename, generate_kwargs={"language": "hebrew"}, return_timestamps=True)
            for chunk in result["chunks"]:
                ts = chunk["timestamp"]
                start, end = ts
                start_s = f"{start:.1f}s" if start is not None else "?"
                end_s = f"{end:.1f}s" if end is not None else "?"
                print(f"[{start_s} - {end_s}] {chunk['text']}")
            return result["text"]

        # def transcribe(filename):
        #     return pipe(filename, generate_kwargs={"language": "hebrew"})["text"]


        return transcribe

    if engine == "faster-whisper":
        model = faster_whisper.WhisperModel(model_path, compute_type=compute_type)
        window_log = apply_faster_whisper_fixes(
            model, max_previous_tokens=max_previous_tokens, fix_seek=fix_seek, log_windows=log_windows
        )

        def transcribe(filename):
            return transcribe_faster_whisper(model, filename, window_log)

        return transcribe


def copy_hf_model_weights_to_openai_model_weights(tuned_model, model):
    dic_parameter_mapping = dict()

    for i in range(32):
        dic_parameter_mapping[f"decoder.blocks.{i}.attn.key.weight"] = (
            f"model.decoder.layers.{i}.self_attn.k_proj.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.attn.out.bias"] = f"model.decoder.layers.{i}.self_attn.out_proj.bias"
        dic_parameter_mapping[f"decoder.blocks.{i}.attn.out.weight"] = (
            f"model.decoder.layers.{i}.self_attn.out_proj.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.attn.query.bias"] = f"model.decoder.layers.{i}.self_attn.q_proj.bias"
        dic_parameter_mapping[f"decoder.blocks.{i}.attn.query.weight"] = (
            f"model.decoder.layers.{i}.self_attn.q_proj.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.attn.value.bias"] = f"model.decoder.layers.{i}.self_attn.v_proj.bias"
        dic_parameter_mapping[f"decoder.blocks.{i}.attn.value.weight"] = (
            f"model.decoder.layers.{i}.self_attn.v_proj.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.attn_ln.bias"] = (
            f"model.decoder.layers.{i}.self_attn_layer_norm.bias"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.attn_ln.weight"] = (
            f"model.decoder.layers.{i}.self_attn_layer_norm.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.cross_attn.key.weight"] = (
            f"model.decoder.layers.{i}.encoder_attn.k_proj.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.cross_attn.out.bias"] = (
            f"model.decoder.layers.{i}.encoder_attn.out_proj.bias"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.cross_attn.out.weight"] = (
            f"model.decoder.layers.{i}.encoder_attn.out_proj.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.cross_attn.query.bias"] = (
            f"model.decoder.layers.{i}.encoder_attn.q_proj.bias"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.cross_attn.query.weight"] = (
            f"model.decoder.layers.{i}.encoder_attn.q_proj.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.cross_attn.value.bias"] = (
            f"model.decoder.layers.{i}.encoder_attn.v_proj.bias"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.cross_attn.value.weight"] = (
            f"model.decoder.layers.{i}.encoder_attn.v_proj.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.cross_attn_ln.bias"] = (
            f"model.decoder.layers.{i}.encoder_attn_layer_norm.bias"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.cross_attn_ln.weight"] = (
            f"model.decoder.layers.{i}.encoder_attn_layer_norm.weight"
        )
        dic_parameter_mapping[f"decoder.blocks.{i}.mlp.0.bias"] = f"model.decoder.layers.{i}.fc1.bias"
        dic_parameter_mapping[f"decoder.blocks.{i}.mlp.0.weight"] = f"model.decoder.layers.{i}.fc1.weight"
        dic_parameter_mapping[f"decoder.blocks.{i}.mlp.2.bias"] = f"model.decoder.layers.{i}.fc2.bias"
        dic_parameter_mapping[f"decoder.blocks.{i}.mlp.2.weight"] = f"model.decoder.layers.{i}.fc2.weight"
        dic_parameter_mapping[f"decoder.blocks.{i}.mlp_ln.bias"] = f"model.decoder.layers.{i}.final_layer_norm.bias"
        dic_parameter_mapping[f"decoder.blocks.{i}.mlp_ln.weight"] = f"model.decoder.layers.{i}.final_layer_norm.weight"

        dic_parameter_mapping[f"encoder.blocks.{i}.attn.key.weight"] = (
            f"model.encoder.layers.{i}.self_attn.k_proj.weight"
        )
        dic_parameter_mapping[f"encoder.blocks.{i}.attn.out.weight"] = (
            f"model.encoder.layers.{i}.self_attn.out_proj.weight"
        )
        dic_parameter_mapping[f"encoder.blocks.{i}.attn.query.weight"] = (
            f"model.encoder.layers.{i}.self_attn.q_proj.weight"
        )
        dic_parameter_mapping[f"encoder.blocks.{i}.attn.value.weight"] = (
            f"model.encoder.layers.{i}.self_attn.v_proj.weight"
        )
        dic_parameter_mapping[f"encoder.blocks.{i}.mlp.2.weight"] = f"model.encoder.layers.{i}.fc2.weight"
        dic_parameter_mapping[f"encoder.blocks.{i}.attn.out.bias"] = f"model.encoder.layers.{i}.self_attn.out_proj.bias"
        dic_parameter_mapping[f"encoder.blocks.{i}.attn.query.bias"] = f"model.encoder.layers.{i}.self_attn.q_proj.bias"
        dic_parameter_mapping[f"encoder.blocks.{i}.attn.value.bias"] = f"model.encoder.layers.{i}.self_attn.v_proj.bias"
        dic_parameter_mapping[f"encoder.blocks.{i}.attn_ln.bias"] = (
            f"model.encoder.layers.{i}.self_attn_layer_norm.bias"
        )
        dic_parameter_mapping[f"encoder.blocks.{i}.attn_ln.weight"] = (
            f"model.encoder.layers.{i}.self_attn_layer_norm.weight"
        )
        dic_parameter_mapping[f"encoder.blocks.{i}.mlp.2.bias"] = f"model.encoder.layers.{i}.fc2.bias"
        dic_parameter_mapping[f"encoder.blocks.{i}.mlp_ln.bias"] = f"model.encoder.layers.{i}.final_layer_norm.bias"
        dic_parameter_mapping[f"encoder.blocks.{i}.mlp_ln.weight"] = f"model.encoder.layers.{i}.final_layer_norm.weight"
        dic_parameter_mapping[f"encoder.blocks.{i}.mlp.0.weight"] = f"model.encoder.layers.{i}.fc1.weight"
        dic_parameter_mapping[f"encoder.blocks.{i}.mlp.0.bias"] = f"model.encoder.layers.{i}.fc1.bias"

    dic_parameter_mapping["encoder.conv1.bias"] = "model.encoder.conv1.bias"
    dic_parameter_mapping["encoder.conv1.weight"] = "model.encoder.conv1.weight"
    dic_parameter_mapping["encoder.conv2.bias"] = "model.encoder.conv2.bias"
    dic_parameter_mapping["encoder.conv2.weight"] = "model.encoder.conv2.weight"
    dic_parameter_mapping["encoder.ln_post.bias"] = "model.encoder.layer_norm.bias"
    dic_parameter_mapping["encoder.ln_post.weight"] = "model.encoder.layer_norm.weight"
    dic_parameter_mapping["encoder.positional_embedding"] = "model.encoder.embed_positions.weight"

    dic_parameter_mapping["decoder.ln.bias"] = "model.decoder.layer_norm.bias"
    dic_parameter_mapping["decoder.ln.weight"] = "model.decoder.layer_norm.weight"
    dic_parameter_mapping["decoder.positional_embedding"] = "model.decoder.embed_positions.weight"
    dic_parameter_mapping["decoder.token_embedding.weight"] = "model.decoder.embed_tokens.weight"

    model_state_dict = model.state_dict()
    tuned_model_state_dict = tuned_model.state_dict()

    for source_param, target_param in dic_parameter_mapping.items():
        model_state_dict[source_param] = tuned_model_state_dict[target_param]

    model.load_state_dict(model_state_dict)

    return model


def transcribe_openai_whisper(model, filename):
    return model.transcribe(filename, language="he", beam_size=5, best_of=5)["text"]


# faster-whisper fixes for text lost at window ends.
# Background and measurements: docs/faster_whisper_missing_text_investigation.md


def limit_previous_tokens(model, max_tokens):
    """Cap the previous-text prompt that faster-whisper feeds to the decoder.

    Decoding stops at max_length (448) counting the prompt. faster-whisper passes up to 223 previous tokens,
    which leaves about 221 output tokens, less than dense Hebrew windows need (training targets reach 270+).
    Training shortens the previous text to fit the target, so do the same here and leave room for the output.
    """
    original_get_prompt = model.get_prompt

    def get_prompt(tokenizer, previous_tokens, *args, **kwargs):
        previous_tokens = previous_tokens[-max_tokens:] if previous_tokens else previous_tokens
        return original_get_prompt(tokenizer, previous_tokens, *args, **kwargs)

    model.get_prompt = get_prompt


def fix_window_seek(model):
    """Start the next window at the last decoded word even when the model ended the window with one timestamp.

    Upstream only moves seek to the last word end when the window did not end with a single closing timestamp.
    Otherwise it jumps a full 30s window, so speech the model left undecoded at the window end is lost.
    Reporting single_timestamp_ending=False lets transcribe.py:1288 set seek to the last aligned word end.
    Requires word_timestamps=True.

    The last window keeps the upstream behavior: nothing follows it, and seeking back only re-decodes the silent
    tail, where the model hallucinates.
    """
    original = model._split_segments_by_timestamps

    def split_segments(*args, **kwargs):
        current_segments, seek, single_timestamp_ending = original(*args, **kwargs)
        is_last_window = kwargs["segment_size"] < model.feature_extractor.nb_max_frames
        return current_segments, seek, single_timestamp_ending and is_last_window

    model._split_segments_by_timestamps = split_segments


class WindowLog:
    """Print one line per decoded window and a summary, to see how each window ended (debugging only).

    ending:
      open       the last segment was left unclosed. Upstream restarts at the last timestamp, nothing is lost.
      single     the model closed its last segment and stopped. Upstream jumps a full window.
      collapsed  no consecutive timestamps (one long segment). Upstream jumps a full window.
    gap is the window end minus the last timestamp token. For collapsed windows only the opening timestamp
    exists, so the gap prints about 30s, which is an artifact.
    budget is the number of output tokens the prompt leaves. A window is cut off when output_tokens reaches it.

    Must wrap the model before fix_window_seek, so it sees the ending the model produced.
    """

    JUMP_ENDINGS = ("single", "collapsed")

    def __init__(self, model):
        self.model = model
        self.entries = []  # (ending, gap, line)
        self.prompt_len = 0

        original_get_prompt = model.get_prompt
        original_split = model._split_segments_by_timestamps

        def get_prompt(*args, **kwargs):
            prompt = original_get_prompt(*args, **kwargs)
            self.prompt_len = len(prompt)
            return prompt

        def split_segments(*args, **kwargs):
            result = original_split(*args, **kwargs)
            self._record(single_timestamp_ending=result[2], **kwargs)
            return result

        model.get_prompt = get_prompt
        model._split_segments_by_timestamps = split_segments

    def _record(self, single_timestamp_ending, tokenizer, tokens, time_offset, segment_duration, **_):
        ts_begin = tokenizer.timestamp_begin
        timestamps = [t for t in tokens if t >= ts_begin]
        has_consecutive = any(a >= ts_begin and b >= ts_begin for a, b in zip(tokens, tokens[1:]))
        if single_timestamp_ending:
            ending = "single"
        elif has_consecutive:
            ending = "open"
        else:
            ending = "collapsed"
        window_end = time_offset + segment_duration
        last_ts = time_offset + (timestamps[-1] - ts_begin) * self.model.time_precision if timestamps else time_offset
        gap = window_end - last_ts
        line = (
            f"[window] start={time_offset:.2f}s end={window_end:.2f}s ending={ending} "
            f"last_ts={last_ts:.2f}s gap_to_window_end={gap:.2f}s output_tokens={len(tokens)} "
            f"prompt_tokens={self.prompt_len} budget={self.model.max_length - self.prompt_len}"
        )
        self.entries.append((ending, gap, line))
        print(line)

    def print_summary(self, gap_threshold=1.0):
        endings = [ending for ending, _, _ in self.entries]
        risky = [gap for ending, gap, _ in self.entries if ending in self.JUMP_ENDINGS and gap > gap_threshold]
        print(
            f"[window summary] windows={len(endings)} single={endings.count('single')} "
            f"open={endings.count('open')} collapsed={endings.count('collapsed')} "
            f"full_jump_with_gap>{gap_threshold:.1f}s={len(risky)}"
        )
        for ending, _, line in self.entries:
            if ending in self.JUMP_ENDINGS:
                print(line)


def apply_faster_whisper_fixes(model, max_previous_tokens=128, fix_seek=True, log_windows=False):
    """Patch a faster_whisper.WhisperModel in place. Returns a WindowLog when log_windows is set, else None."""
    if max_previous_tokens:
        limit_previous_tokens(model, max_previous_tokens)
    window_log = WindowLog(model) if log_windows else None
    if fix_seek:
        fix_window_seek(model)
    return window_log


def transcribe_faster_whisper(model, filename, window_log=None):
    if window_log is not None:
        logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
        window_log.entries.clear()
    texts = []
    segs, info = model.transcribe(
        filename,
        language="he",
        # condition_on_previous_text=True,
        vad_filter=True,
        word_timestamps=True,  # required by fix_window_seek
        compression_ratio_threshold=2.6,
        # clip_timestamps="226.88,300"
    )
    for s in segs:
        if window_log is not None:
            print(
                f"[{s.start:.1f}s - {s.end:.1f}s] (seek={s.seek}, temp={s.temperature}, no_speech={s.no_speech_prob:.3f}, "
                f"avg_logprob={s.avg_logprob:.3f}, compression={s.compression_ratio:.2f}) {s.text}"
            )
        texts.append(s.text)
    if window_log is not None:
        window_log.print_summary()
    # for s in segs:
    #     print(f"{s.text}")
    #     texts.append(s.text)

    return " ".join(texts)


if __name__ == "__main__":
    # Define an argument parser
    parser = argparse.ArgumentParser(description="Create a dataset and upload to Huggingface.")

    # Add the arguments
    parser.add_argument(
        "--engine",
        type=str,
        required=True,
        choices={"openai-whisper", "openai-whisper-tuned", "transformers", "faster-whisper"},
        help="Engine to use.",
    )
    parser.add_argument("--model", type=str, required=True, help="Model to use, (large-v2, large-v3 etc).")
    parser.add_argument(
        "--tuned-model", type=str, required=False, help="Huggingface-style tuned model. Can be hosted or local."
    )
    parser.add_argument("--file", type=str, required=True, help="File to transcribe.")
    parser.add_argument("--compute-type", type=str, default="float16", help="Compute type (float16, float32, int8).")
    parser.add_argument(
        "--max-previous-tokens",
        type=int,
        default=128,
        help="faster-whisper: max previous-text tokens in the prompt, leaves room for the output. 0 keeps the default 223.",
    )
    parser.add_argument(
        "--no-seek-fix",
        action="store_true",
        help="faster-whisper: do not restart the next window at the last decoded word after a single-timestamp ending.",
    )
    parser.add_argument(
        "--log-windows",
        action="store_true",
        help="faster-whisper: print per-window ending, token budget and segments, plus faster_whisper debug logs.",
    )

    # Parse the arguments
    args = parser.parse_args()

    print(f"Loading model {args.model}, tuned model is {args.tuned_model}...")
    transcribe_fn = initialize_model(
        args.engine,
        args.model,
        args.tuned_model,
        getattr(args, "compute_type", "float16"),
        max_previous_tokens=args.max_previous_tokens,
        fix_seek=not args.no_seek_fix,
        log_windows=args.log_windows,
    )

    print(f"Transcribing {args.file}...")
    text = transcribe_fn(args.file)
    print(text)
    open("x.txt", "w").write(text)
