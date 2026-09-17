from .nodes import (
    AudioSpectralPerturbation,
    AudioMicroTimeStretch,
    AudioCodecReencode,
    AudioFingerprintPipeline,
    AudioAdvancedPhaseScrambling,
    AudioMidSidePerturbation,
    AudioPsychoacousticNoiseShaping,
    AudioMultiCodecChain,
    AudioTemporalMicroEdit,
    AudioAdaptiveParameterSelector,
    AudioSpectralEnvelopeWarping,
    AudioFingerprintPipelineV2,
)

NODE_CLASS_MAPPINGS = {
    "AudioSpectralPerturbation": AudioSpectralPerturbation,
    "AudioMicroTimeStretch": AudioMicroTimeStretch,
    "AudioCodecReencode": AudioCodecReencode,
    "AudioFingerprintPipeline": AudioFingerprintPipeline,
    "AudioAdvancedPhaseScrambling": AudioAdvancedPhaseScrambling,
    "AudioMidSidePerturbation": AudioMidSidePerturbation,
    "AudioPsychoacousticNoiseShaping": AudioPsychoacousticNoiseShaping,
    "AudioMultiCodecChain": AudioMultiCodecChain,
    "AudioTemporalMicroEdit": AudioTemporalMicroEdit,
    "AudioAdaptiveParameterSelector": AudioAdaptiveParameterSelector,
    "AudioSpectralEnvelopeWarping": AudioSpectralEnvelopeWarping,
    "AudioFingerprintPipelineV2": AudioFingerprintPipelineV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioSpectralPerturbation": "Audio Spectral Perturbation",
    "AudioMicroTimeStretch": "Audio Micro Time-Stretch",
    "AudioCodecReencode": "Audio Codec Re-encode",
    "AudioFingerprintPipeline": "Audio Fingerprint Remover (Full Pipeline)",
    "AudioAdvancedPhaseScrambling": "Audio Advanced Phase Scrambling",
    "AudioMidSidePerturbation": "Audio Mid/Side Perturbation",
    "AudioPsychoacousticNoiseShaping": "Audio Psychoacoustic Noise Shaping",
    "AudioMultiCodecChain": "Audio Multi-Codec Chain",
    "AudioTemporalMicroEdit": "Audio Temporal Micro-Editing",
    "AudioAdaptiveParameterSelector": "Audio Adaptive Parameter Selector",
    "AudioSpectralEnvelopeWarping": "Audio Spectral Envelope Warping",
    "AudioFingerprintPipelineV2": "Audio Fingerprint Remover (Enhanced Pipeline)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]