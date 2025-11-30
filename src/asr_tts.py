from typing import Tuple
import numpy as np

from faster_whisper import WhisperModel

try:
    from f5_tts import F5TTS
except ImportError:
    F5TTS = None  # Allow running without TTS for text-only mode

from .config import ASR_MODEL_NAME


class ASRWrapper:
    """Simple wrapper around faster-whisper for speech-to-text."""

    def __init__(self, model_name: str = ASR_MODEL_NAME, device: str = "auto"):
        self.model = WhisperModel(model_name, device=device)

    def speech_to_text(self, audio: Tuple[int, np.ndarray]) -> str:
        sr, data = audio
        segments, _ = self.model.transcribe(data, sampling_rate=sr)
        text = " ".join([seg.text.strip() for seg in segments])
        return text.strip()


class TTSWrapper:
    """Wrapper around F5-TTS. Falls back to None if not available."""

    def __init__(self, device: str = "auto"):
        if F5TTS is None:
            self.model = None
        else:
            self.model = F5TTS.from_pretrained("f5-tts-mini")

    def text_to_speech(self, text: str):
        if self.model is None:
            # For setups without TTS installed, return silent audio (or raise)
            # Here we generate 0.5 s of silence at 22.05 kHz
            sr = 22050
            audio = np.zeros(int(0.5 * sr), dtype=np.float32)
            return sr, audio

        audio = self.model.speak(text, voice="neutral")
        sr = 22050  # F5-TTS default
        return sr, audio