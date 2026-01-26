# core/asr.py
from __future__ import annotations
from typing import Tuple, Any
import numpy as np
import soundfile as sf
import torch
import torchaudio


def read_wav_resample(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    if sr != target_sr:
        t = torch.from_numpy(audio).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, target_sr)
        audio = t.squeeze(0).numpy()
        sr = target_sr

    return audio, sr


def transcribe_audio(
    asr_pipe: Any,
    wav_path: str,
    target_sr: int = 16000,
    language: str = "indonesian",
) -> str:
    audio, sr = read_wav_resample(wav_path, target_sr)
    result = asr_pipe(
        {"array": audio, "sampling_rate": sr},
        generate_kwargs={"task": "transcribe", "language": language},
        chunk_length_s=20,
        stride_length_s=3,
    )
    return result["text"].strip()
