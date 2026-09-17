from .nodes import (
    AudioSpectralPerturbation,
    AudioMicroTimeStretch,
    AudioCodecReencode,
    AudioFingerprintPipeline,
)

NODE_CLASS_MAPPINGS = {
    "AudioSpectralPerturbation": AudioSpectralPerturbation,
    "AudioMicroTimeStretch": AudioMicroTimeStretch,
    "AudioCodecReencode": AudioCodecReencode,
    "AudioFingerprintPipeline": AudioFingerprintPipeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioSpectralPerturbation": "Audio Spectral Perturbation",
    "AudioMicroTimeStretch": "Audio Micro Time-Stretch",
    "AudioCodecReencode": "Audio Codec Re-encode",
    "AudioFingerprintPipeline": "Audio Fingerprint Remover (Full Pipeline)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]