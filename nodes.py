"""
ComfyUI-AudioReEncoder
Removes codec-level and spectral fingerprints from AI-generated audio
so that downstream classifiers do not attribute the output to a
specific upstream generator.

All processing is deterministic given a fixed seed and operates on
the ComfyUI AUDIO type: {"waveform": Tensor (B, C, T), "sample_rate": int}
"""

import torch
import numpy as np
import math
import os
import subprocess
import tempfile
import shutil
import logging
import random
from typing import Optional, Tuple, List

logger = logging.getLogger("AudioFingerprintRemover")


# ──────────────────────────────────────────────
#  Advanced Helpers
# ──────────────────────────────────────────────

def _hz_to_erb(freqs: torch.Tensor) -> torch.Tensor:
    """Convert Hz to ERB-rate scale."""
    return 21.4 * torch.log10(1 + freqs / 4500.0)


def _erb_to_hz(erb: torch.Tensor) -> torch.Tensor:
    """Convert ERB-rate scale back to Hz."""
    return 4500.0 * (10.0 ** (erb / 21.4) - 1.0)


def _spreading_function(center_erb: torch.Tensor, target_erb: torch.Tensor,
                        is_suppression: bool = True) -> torch.Tensor:
    """ERB-domain spreading function for masking calculations."""
    delta = target_erb - center_erb
    if is_suppression:
        spread = torch.where(delta < 0,
                             27.0 * delta,
                             -12.0 * delta)
    else:
        spread = -24.0 * torch.abs(delta)
    return 10.0 ** (spread / 10.0)


def _compute_masking_threshold(magnitude: torch.Tensor, freqs: torch.Tensor,
                               sr: int, n_fft: int) -> torch.Tensor:
    """
    Compute psychoacoustic masking threshold per frequency bin.
    magnitude: (C, F, frames)
    Returns threshold in linear magnitude scale, shape (C, F, frames).
    """
    C, F, frames = magnitude.shape
    device = magnitude.device

    erb_freqs = _hz_to_erb(freqs)
    n_bands = 40
    erb_min, erb_max = erb_freqs[0].item(), erb_freqs[-1].item()
    band_edges = torch.linspace(erb_min, erb_max, n_bands + 1, device=device)
    band_centers = (band_edges[:-1] + band_edges[1:]) / 2

    band_energy = torch.zeros(C, n_bands, device=device)
    for b in range(n_bands):
        mask = (erb_freqs >= band_edges[b]) & (erb_freqs < band_edges[b + 1])
        if mask.any():
            band_energy[:, b] = magnitude[:, mask, :].mean(dim=(1, 2))

    spreading = torch.zeros(n_bands, n_bands, device=device)
    for i in range(n_bands):
        spreading[i] = _spreading_function(band_centers[i], band_centers, True)

    masked_energy = torch.matmul(band_energy, spreading.T)  # (C, n_bands)

    threshold = torch.zeros(C, F, device=device)
    for b in range(n_bands):
        mask = (erb_freqs >= band_edges[b]) & (erb_freqs < band_edges[b + 1])
        if mask.any():
            threshold[:, mask] = masked_energy[:, b:b + 1]

    # Absolute threshold of hearing — clamp freqs to avoid inf at 0 Hz
    safe_freqs = torch.clamp(freqs, min=1.0)
    ath_db = (3.64 * (safe_freqs / 1000) ** -0.8
              - 6.5 * torch.exp(-0.6 * (safe_freqs / 1000 - 3.3) ** 2)
              + 1e-3 * (safe_freqs / 1000) ** 4)
    ath = 10.0 ** (ath_db / 20.0)  # dB → linear
    ath = ath.to(device)

    # Element-wise max: (C, F) vs (1, F) → (C, F)
    threshold = torch.maximum(threshold, ath.unsqueeze(0))

    # Expand to match frames dimension
    return threshold.unsqueeze(-1).expand(-1, -1, frames)


def _stereo_to_mid_side(wav: torch.Tensor) -> torch.Tensor:
    """Convert stereo (2, T) to mid/side (2, T)."""
    if wav.shape[0] != 2:
        return wav
    mid = (wav[0] + wav[1]) * 0.5
    side = (wav[0] - wav[1]) * 0.5
    return torch.stack([mid, side], dim=0)


def _mid_side_to_stereo(wav: torch.Tensor) -> torch.Tensor:
    """Convert mid/side (2, T) to stereo (2, T)."""
    if wav.shape[0] != 2:
        return wav
    left = wav[0] + wav[1]
    right = wav[0] - wav[1]
    return torch.stack([left, right], dim=0)


# ──────────────────────────────────────────────
#  Core Helpers
# ──────────────────────────────────────────────

def _validate_audio(audio: dict) -> tuple:
    """Return (waveform, sample_rate) with shape guarantees."""
    wav = audio["waveform"]
    sr = int(audio["sample_rate"])
    if wav.dim() == 2:
        wav = wav.unsqueeze(0)
    if wav.dim() != 3:
        raise ValueError("audio waveform must have shape [batch, channels, samples]")
    return wav, sr


def _make_audio(wav: torch.Tensor, sr: int) -> dict:
    return {"waveform": wav, "sample_rate": sr}


def _stft(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    """x: (C, T) → complex (C, F, frames)"""
    window = torch.hann_window(n_fft, device=x.device)
    return torch.stft(x, n_fft=n_fft, hop_length=hop,
                      window=window, return_complex=True,
                      pad_mode="reflect")


def _istft(S: torch.Tensor, n_fft: int, hop: int, length: int) -> torch.Tensor:
    """S: complex (C, F, frames) → (C, T)"""
    window = torch.hann_window(n_fft, device=S.device)
    return torch.istft(S, n_fft=n_fft, hop_length=hop,
                       window=window, length=length)


# ──────────────────────────────────────────────
#  1. Spectral Perturbation
# ──────────────────────────────────────────────

class AudioSpectralPerturbation:
    """
    Applies imperceptible spectral perturbations that break the
    phase-coherence and noise-floor signatures used by AI-audio
    detectors, without audible quality loss.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "phase_jitter_strength": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Random phase rotation applied above the cutoff. "
                               "0.15-0.35 is inaudible; >0.5 causes phasing."
                }),
                "jitter_cutoff_hz": ("FLOAT", {
                    "default": 8000.0, "min": 2000.0, "max": 20000.0,
                    "step": 100.0,
                    "tooltip": "Only frequencies above this value receive jitter. "
                               "Keep >= 7000 to stay inaudible."
                }),
                "dither_db": ("FLOAT", {
                    "default": -65.0, "min": -90.0, "max": -40.0,
                    "step": 1.0,
                    "tooltip": "Broadband noise floor in dBFS. "
                               "-65 is well below audibility."
                }),
                "tilt_db": ("FLOAT", {
                    "default": 0.0, "min": -3.0, "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Broad high-shelf tilt in dB. "
                               "+/-1 dB is inaudible; shifts spectral envelope."
                }),
                "tilt_freq_hz": ("FLOAT", {
                    "default": 10000.0, "min": 2000.0, "max": 20000.0,
                    "step": 500.0,
                    "tooltip": "Centre frequency of the spectral tilt shelf."
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2**31,
                    "tooltip": "0 = random every run. Fix for reproducibility."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "process"
    CATEGORY = "audio/fingerprint_removal"

    def process(self, audio, phase_jitter_strength, jitter_cutoff_hz,
                dither_db, tilt_db, tilt_freq_hz, seed):
        wav, sr = _validate_audio(audio)
        B, C, T = wav.shape

        if seed == 0:
            seed = torch.seed()
        gen = torch.Generator(device=wav.device).manual_seed(seed)

        n_fft = 2048
        hop = n_fft // 4
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr, device=wav.device)

        out = torch.empty_like(wav)

        for b in range(B):
            for c in range(C):
                sig = wav[b, c]
                S = _stft(sig.unsqueeze(0), n_fft, hop).squeeze(0)
                mag = S.abs()
                phase = S.angle()

                if phase_jitter_strength > 0:
                    mask = (freqs >= jitter_cutoff_hz).float().to(phase.device)
                    noise = (torch.rand(phase.shape, device=phase.device,
                                        generator=gen) * 2 * math.pi) - math.pi
                    phase = phase + noise * mask.unsqueeze(-1) * phase_jitter_strength

                if abs(tilt_db) > 0.01:
                    shelf = torch.sigmoid(
                        (freqs - tilt_freq_hz) / (tilt_freq_hz * 0.25)
                    ).to(mag.device)
                    gain_lin = 10.0 ** (tilt_db / 20.0)
                    gain_curve = 1.0 + shelf * (gain_lin - 1.0)
                    mag = mag * gain_curve.unsqueeze(-1)

                S_out = mag * torch.exp(1j * phase)
                sig_out = _istft(S_out.unsqueeze(0), n_fft, hop, T).squeeze(0)

                if dither_db > -90:
                    amp = 10.0 ** (dither_db / 20.0)
                    sig_out = sig_out + torch.randn(
                        sig_out.shape, device=sig_out.device,
                        generator=gen) * amp

                out[b, c] = sig_out

        out = torch.clamp(out, -1.0, 1.0)
        return (_make_audio(out, sr),)


# ──────────────────────────────────────────────
#  2. Micro Time-Stretch (pitch-preserving)
# ──────────────────────────────────────────────

class AudioMicroTimeStretch:
    """
    Phase-vocoder time-stretch by +/-0.1 - 2 %.
    Shifts temporal statistics without changing pitch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "stretch_percent": ("FLOAT", {
                    "default": 0.5, "min": -2.0, "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Percentage of time-stretch. "
                               "+/-0.3-0.8%% is inaudible."
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2**31,
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "process"
    CATEGORY = "audio/fingerprint_removal"

    def process(self, audio, stretch_percent, seed):
        wav, sr = _validate_audio(audio)
        if abs(stretch_percent) < 0.01:
            return (audio,)

        rate = 1.0 + stretch_percent / 100.0
        B, C, T = wav.shape

        n_fft = 2048
        hop = n_fft // 4
        out_batches = []

        for b in range(B):
            out_channels = []
            for c in range(C):
                sig = wav[b, c].unsqueeze(0)
                S = torch.stft(sig, n_fft, hop_length=hop,
                               window=torch.hann_window(n_fft, device=sig.device),
                               return_complex=True, pad_mode="reflect")
                S = S.squeeze(0)  # (F, frames)

                n_frames_in = S.shape[-1]
                n_frames_out = int(n_frames_in / rate)

                mag_in = S.abs()
                phase_in = S.angle()

                mag_out = torch.nn.functional.interpolate(
                    mag_in.unsqueeze(0).unsqueeze(0),
                    size=(mag_in.shape[0], n_frames_out),
                    mode="bilinear", align_corners=False
                ).squeeze(0).squeeze(0)

                # Unwrap phase along time
                phase_delta = phase_in[..., 1:] - phase_in[..., :-1]
                wrapped_delta = (phase_delta + math.pi) % (2 * math.pi) - math.pi
                phase_unwrapped = torch.cat(
                    (phase_in[..., :1],
                     phase_in[..., :1] + torch.cumsum(wrapped_delta, dim=-1)),
                    dim=-1
                )

                phase_out = torch.nn.functional.interpolate(
                    phase_unwrapped.unsqueeze(0).unsqueeze(0),
                    size=(phase_unwrapped.shape[0], n_frames_out),
                    mode="bilinear", align_corners=False
                ).squeeze(0).squeeze(0)

                S_out = mag_out * torch.exp(1j * phase_out)
                new_len = int(T / rate)
                sig_out = torch.istft(
                    S_out.unsqueeze(0), n_fft, hop_length=hop,
                    window=torch.hann_window(n_fft, device=sig.device),
                    length=new_len
                ).squeeze(0)

                out_channels.append(sig_out)

            out_batches.append(torch.stack(out_channels, dim=0))

        w_out = torch.stack(out_batches, dim=0)
        w_out = torch.clamp(w_out, -1.0, 1.0)
        return (_make_audio(w_out, sr),)


# ──────────────────────────────────────────────
#  3. Codec Re-encode (FFmpeg round-trip)
# ──────────────────────────────────────────────

class AudioCodecReencode:
    """
    Encodes audio to a lossy codec and decodes it back,
    destroying codec-level reconstruction fingerprints.
    Requires FFmpeg on the system PATH.
    """

    CODECS = ["opus", "aac", "libmp3lame", "flac"]
    CODEC_ENCODERS = {"opus": "libopus", "aac": "aac",
                      "libmp3lame": "libmp3lame", "flac": "flac"}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "codec": (cls.CODECS, {
                    "default": "opus",
                    "tooltip": "Opus at 256-320k is transparent and "
                               "destroys neural-codec fingerprints."
                }),
                "bitrate_kbps": ("INT", {
                    "default": 320, "min": 64, "max": 512, "step": 32,
                    "tooltip": "Encoder bitrate. >= 256 recommended."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "process"
    CATEGORY = "audio/fingerprint_removal"

    def process(self, audio, codec, bitrate_kbps):
        wav, sr = _validate_audio(audio)

        if shutil.which("ffmpeg") is None:
            logger.warning("ffmpeg not found - skipping codec re-encode.")
            return (audio,)

        B, C, T = wav.shape
        results = []

        for b in range(B):
            w = wav[b]  # (C, T)
            with tempfile.TemporaryDirectory() as tmp:
                src = os.path.join(tmp, "src.wav")
                enc = os.path.join(tmp, f"enc.{self._ext(codec)}")
                dec = os.path.join(tmp, "dec.wav")

                raw_audio = (
                    w.detach().float().cpu().clamp(-1.0, 1.0).numpy()
                    .T.astype(np.float32, copy=False).tobytes()
                )

                cmd_src = [
                    "ffmpeg", "-y",
                    "-f", "f32le", "-ar", str(sr), "-ac", str(C),
                    "-i", "pipe:0", "-c:a", "pcm_s16le", src,
                ]
                subprocess.run(cmd_src, input=raw_audio,
                               capture_output=True, check=True)

                cmd_enc = [
                    "ffmpeg", "-y", "-i", src,
                    "-c:a", self.CODEC_ENCODERS[codec],
                ]
                if codec != "flac":
                    effective_bitrate = min(bitrate_kbps, 256) if codec == "opus" else bitrate_kbps
                    cmd_enc.extend(["-b:a", f"{effective_bitrate}k"])
                cmd_enc.append(enc)
                subprocess.run(cmd_enc, capture_output=True, check=True)

                cmd_dec_wav = ["ffmpeg", "-y", "-i", enc, dec]
                subprocess.run(cmd_dec_wav, capture_output=True, check=True)

                cmd_pipe = [
                    "ffmpeg", "-v", "error", "-i", dec,
                    "-f", "f32le", "-acodec", "pcm_f32le",
                    "-ar", str(sr), "-ac", str(C), "pipe:1",
                ]
                decoded = subprocess.run(cmd_pipe, capture_output=True, check=True)
                decoded_np = np.frombuffer(decoded.stdout, dtype=np.float32)

                if decoded_np.size % C:
                    raise RuntimeError("decoded audio has an incomplete sample frame")

                w_dec = torch.from_numpy(decoded_np.reshape(-1, C).T.copy())

                if w_dec.shape[0] != C:
                    raise RuntimeError(
                        f"decoded audio has {w_dec.shape[0]} channels; expected {C}")

                if w_dec.shape[1] < T:
                    w_dec = torch.nn.functional.pad(w_dec, (0, T - w_dec.shape[1]))
                w_dec = w_dec[:, :T]
                results.append(w_dec)

        wav_out = torch.stack(results, dim=0).to(wav.device)
        return (_make_audio(wav_out, sr),)

    @staticmethod
    def _ext(codec: str) -> str:
        return {"opus": "ogg", "aac": "m4a",
                "libmp3lame": "mp3", "flac": "flac"}.get(codec, "ogg")


# ──────────────────────────────────────────────
#  4. Full Pipeline (v1)
# ──────────────────────────────────────────────

class AudioFingerprintPipeline:
    """
    Chains: Spectral Perturbation -> Micro Time-Stretch -> Codec Re-encode.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "phase_jitter_strength": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01
                }),
                "jitter_cutoff_hz": ("FLOAT", {
                    "default": 8000.0, "min": 2000.0, "max": 20000.0, "step": 100.0
                }),
                "dither_db": ("FLOAT", {
                    "default": -65.0, "min": -90.0, "max": -40.0, "step": 1.0
                }),
                "tilt_db": ("FLOAT", {
                    "default": 0.0, "min": -3.0, "max": 3.0, "step": 0.1
                }),
                "stretch_percent": ("FLOAT", {
                    "default": 0.5, "min": -2.0, "max": 2.0, "step": 0.1
                }),
                "enable_codec_reencode": ("BOOLEAN", {"default": True}),
                "codec": (AudioCodecReencode.CODECS, {"default": "opus"}),
                "bitrate_kbps": ("INT", {
                    "default": 320, "min": 64, "max": 512, "step": 32
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "process"
    CATEGORY = "audio/fingerprint_removal"

    def process(self, audio, phase_jitter_strength, jitter_cutoff_hz,
                dither_db, tilt_db, stretch_percent,
                enable_codec_reencode, codec, bitrate_kbps, seed):
        spec_node = AudioSpectralPerturbation()
        (audio,) = spec_node.process(
            audio, phase_jitter_strength, jitter_cutoff_hz,
            dither_db, tilt_db, 10000.0, seed
        )

        if abs(stretch_percent) > 0.01:
            ts_node = AudioMicroTimeStretch()
            (audio,) = ts_node.process(audio, stretch_percent, seed)

        if enable_codec_reencode:
            ce_node = AudioCodecReencode()
            (audio,) = ce_node.process(audio, codec, bitrate_kbps)

        return (audio,)


# ──────────────────────────────────────────────
#  5. Advanced Phase Scrambling (FIXED)
# ──────────────────────────────────────────────

class AudioAdvancedPhaseScrambling:
    """
    Correlation-preserving phase scrambling that breaks detector
    phase-coherence signatures. Uses ERB-scale correlated phase
    perturbations with proper inter-channel correlation control.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "strength": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Scrambling strength. 0.2-0.4 is transparent."
                }),
                "correlation_preservation": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Preserve inter-channel phase correlation. "
                               "Higher = more natural stereo image."
                }),
                "erb_smoothing": ("FLOAT", {
                    "default": 1.5, "min": 0.5, "max": 5.0,
                    "step": 0.1,
                    "tooltip": "ERB-domain smoothing for natural phase evolution."
                }),
                "min_freq_hz": ("FLOAT", {
                    "default": 2000.0, "min": 500.0, "max": 8000.0,
                    "step": 100.0,
                    "tooltip": "Minimum frequency for scrambling."
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2**31,
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "process"
    CATEGORY = "audio/fingerprint_removal"

    def process(self, audio, strength, correlation_preservation, erb_smoothing,
                min_freq_hz, seed):
        wav, sr = _validate_audio(audio)
        B, C, T = wav.shape

        if seed == 0:
            seed = torch.seed()
        gen = torch.Generator(device=wav.device).manual_seed(seed)

        n_fft = 2048
        hop = n_fft // 4
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr, device=wav.device)
        erb_freqs = _hz_to_erb(freqs)

        n_erb_bins = 64
        erb_min, erb_max = erb_freqs[0].item(), erb_freqs[-1].item()
        erb_centers = torch.linspace(erb_min, erb_max, n_erb_bins, device=wav.device)

        # Precompute ERB → frequency-bin weights (same for all channels)
        erb_weights = torch.exp(
            -((erb_freqs.unsqueeze(1) - erb_centers.unsqueeze(0)) ** 2)
            / (2 * erb_smoothing ** 2)
        )
        erb_weights = erb_weights / erb_weights.sum(dim=1, keepdim=True)

        freq_mask = (freqs >= min_freq_hz).float().unsqueeze(-1)

        out = torch.empty_like(wav)

        for b in range(B):
            # Determine number of STFT frames from first channel
            test_S = _stft(wav[b, 0].unsqueeze(0), n_fft, hop).squeeze(0)
            n_frames = test_S.shape[-1]

            # Generate base phase noise (reference for inter-channel correlation)
            erb_phase_base = (
                torch.rand(n_erb_bins, n_frames, device=wav.device, generator=gen)
                * 2 - 1
            ) * math.pi
            # Smooth in time
            erb_phase_base = torch.nn.functional.avg_pool1d(
                erb_phase_base.unsqueeze(0), kernel_size=3, stride=1, padding=1
            ).squeeze(0)

            for c in range(C):
                sig = wav[b, c]
                S = _stft(sig.unsqueeze(0), n_fft, hop).squeeze(0)
                mag = S.abs()
                phase = S.angle()

                if c == 0:
                    erb_phase = erb_phase_base
                else:
                    # Independent noise blended with base for correlation control
                    erb_phase_ind = (
                        torch.rand(n_erb_bins, n_frames, device=wav.device, generator=gen)
                        * 2 - 1
                    ) * math.pi
                    erb_phase_ind = torch.nn.functional.avg_pool1d(
                        erb_phase_ind.unsqueeze(0), kernel_size=3, stride=1, padding=1
                    ).squeeze(0)
                    erb_phase = (correlation_preservation * erb_phase_base
                                 + (1.0 - correlation_preservation) * erb_phase_ind)

                # Map ERB-domain noise to frequency bins
                phase_noise = torch.matmul(erb_weights, erb_phase)

                # Apply only above min_freq_hz
                phase = phase + phase_noise * freq_mask * strength

                S_out = mag * torch.exp(1j * phase)
                sig_out = _istft(S_out.unsqueeze(0), n_fft, hop, T).squeeze(0)
                out[b, c] = sig_out

        out = torch.clamp(out, -1.0, 1.0)
        return (_make_audio(out, sr),)


# ──────────────────────────────────────────────
#  6. Mid/Side Stereo Perturbation
# ──────────────────────────────────────────────

class AudioMidSidePerturbation:
    """
    Breaks stereo correlation fingerprints by applying independent
    perturbations to mid and side channels.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "mid_jitter_strength": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 0.5, "step": 0.01,
                }),
                "side_jitter_strength": ("FLOAT", {
                    "default": 0.35, "min": 0.0, "max": 0.7, "step": 0.01,
                }),
                "mid_dither_db": ("FLOAT", {
                    "default": -70.0, "min": -90.0, "max": -50.0, "step": 1.0,
                }),
                "side_dither_db": ("FLOAT", {
                    "default": -60.0, "min": -90.0, "max": -40.0, "step": 1.0,
                }),
                "width_modulation": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 0.2, "step": 0.01,
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2**31,
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "process"
    CATEGORY = "audio/fingerprint_removal"

    def process(self, audio, mid_jitter_strength, side_jitter_strength,
                mid_dither_db, side_dither_db, width_modulation, seed):
        wav, sr = _validate_audio(audio)
        B, C, T = wav.shape

        if C < 2:
            return (audio,)

        if seed == 0:
            seed = torch.seed()
        gen = torch.Generator(device=wav.device).manual_seed(seed)

        n_fft = 2048
        hop = n_fft // 4
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr, device=wav.device)
        erb_freqs = _hz_to_erb(freqs)
        n_bands = 32
        erb_centers = torch.linspace(erb_freqs[0], erb_freqs[-1], n_bands,
                                     device=wav.device)

        out = torch.empty_like(wav)

        for b in range(B):
            ms = _stereo_to_mid_side(wav[b])
            ms_out = torch.empty_like(ms)

            for c in range(2):
                sig = ms[c]
                S = _stft(sig.unsqueeze(0), n_fft, hop).squeeze(0)
                mag = S.abs()
                phase = S.angle()

                jitter_strength = mid_jitter_strength if c == 0 else side_jitter_strength

                if jitter_strength > 0:
                    n_frames = phase.shape[-1]
                    erb_phase = (
                        torch.rand(n_bands, n_frames, device=wav.device, generator=gen)
                        * 2 - 1
                    ) * math.pi
                    erb_weights = torch.exp(
                        -((erb_freqs.unsqueeze(1) - erb_centers.unsqueeze(0)) ** 2) / 8.0
                    )
                    erb_weights = erb_weights / erb_weights.sum(dim=1, keepdim=True)
                    phase_noise = torch.matmul(erb_weights, erb_phase)
                    freq_mask = (freqs >= 3000).float().unsqueeze(-1)
                    phase = phase + phase_noise * freq_mask * jitter_strength

                S_out = mag * torch.exp(1j * phase)
                sig_out = _istft(S_out.unsqueeze(0), n_fft, hop, T).squeeze(0)

                dither_db = mid_dither_db if c == 0 else side_dither_db
                if dither_db > -90:
                    amp = 10.0 ** (dither_db / 20.0)
                    sig_out = sig_out + torch.randn(
                        sig_out.shape, device=wav.device, generator=gen) * amp

                ms_out[c] = sig_out

            if width_modulation > 0:
                t = torch.arange(T, device=wav.device, dtype=torch.float32) / sr
                mod = 1.0 + 0.1 * torch.sin(2 * math.pi * width_modulation * t)
                mod = mod + torch.randn(1, device=wav.device, generator=gen) * 0.02
                ms_out[1] = ms_out[1] * mod

            stereo = _mid_side_to_stereo(ms_out)
            out[b] = stereo

        out = torch.clamp(out, -1.0, 1.0)
        return (_make_audio(out, sr),)


# ──────────────────────────────────────────────
#  7. Psychoacoustic Noise Shaping (FIXED)
# ──────────────────────────────────────────────

class AudioPsychoacousticNoiseShaping:
    """
    Adds dither shaped by psychoacoustic masking thresholds.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_snr_db": ("FLOAT", {
                    "default": 40.0, "min": 20.0, "max": 80.0, "step": 1.0,
                }),
                "masking_margin_db": ("FLOAT", {
                    "default": 6.0, "min": 0.0, "max": 20.0, "step": 0.5,
                }),
                "noise_floor_db": ("FLOAT", {
                    "default": -90.0, "min": -120.0, "max": -60.0, "step": 1.0,
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2**31,
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "process"
    CATEGORY = "audio/fingerprint_removal"

    def process(self, audio, target_snr_db, masking_margin_db, noise_floor_db, seed):
        wav, sr = _validate_audio(audio)
        B, C, T = wav.shape

        if seed == 0:
            seed = torch.seed()
        gen = torch.Generator(device=wav.device).manual_seed(seed)

        n_fft = 2048
        hop = n_fft // 4
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr, device=wav.device)

        out = torch.empty_like(wav)

        for b in range(B):
            for c in range(C):
                sig = wav[b, c]
                S = _stft(sig.unsqueeze(0), n_fft, hop).squeeze(0)
                mag = S.abs()

                # Compute masking threshold
                threshold = _compute_masking_threshold(
                    mag.unsqueeze(0), freqs, sr, n_fft)
                threshold = threshold.squeeze(0)  # (F, frames)

                # Target noise level
                noise_target_db = torch.clamp(
                    20 * torch.log10(threshold + 1e-10) - masking_margin_db,
                    max=noise_floor_db
                )
                noise_target = 10.0 ** (noise_target_db / 20.0)

                # Generate shaped noise in frequency domain
                noise_spec = torch.randn_like(S, generator=gen) * noise_target

                S_out = S + noise_spec
                sig_out = _istft(S_out.unsqueeze(0), n_fft, hop, T).squeeze(0)
                out[b, c] = sig_out

        out = torch.clamp(out, -1.0, 1.0)
        return (_make_audio(out, sr),)


# ──────────────────────────────────────────────
#  8. Multi-Codec Chain
# ──────────────────────────────────────────────

class AudioMultiCodecChain:
    """
    Round-trips audio through multiple different lossy codecs.
    """

    CODEC_CHAINS = {
        "opus_to_aac": ["opus", "aac"],
        "opus_to_mp3": ["opus", "libmp3lame"],
        "aac_to_mp3": ["aac", "libmp3lame"],
        "opus_to_aac_to_mp3": ["opus", "aac", "libmp3lame"],
    }
    CODEC_ENCODERS = {"opus": "libopus", "aac": "aac", "libmp3lame": "libmp3lame"}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "chain": (list(cls.CODEC_CHAINS.keys()), {
                    "default": "opus_to_aac",
                }),
                "bitrate_kbps": ("INT", {
                    "default": 256, "min": 96, "max": 512, "step": 32,
                }),
                "final_bitrate_kbps": ("INT", {
                    "default": 320, "min": 128, "max": 512, "step": 32,
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "process"
    CATEGORY = "audio/fingerprint_removal"

    def process(self, audio, chain, bitrate_kbps, final_bitrate_kbps):
        wav, sr = _validate_audio(audio)

        if shutil.which("ffmpeg") is None:
            logger.warning("ffmpeg not found - skipping multi-codec chain.")
            return (audio,)

        codecs = self.CODEC_CHAINS[chain]
        B, C, T = wav.shape
        current_wav = wav

        for i, codec in enumerate(codecs):
            is_last = (i == len(codecs) - 1)
            br = final_bitrate_kbps if is_last else bitrate_kbps
            results = []

            for b in range(B):
                w = current_wav[b]
                with tempfile.TemporaryDirectory() as tmp:
                    src = os.path.join(tmp, f"src_{i}.wav")
                    enc = os.path.join(tmp, f"enc_{i}.{self._ext(codec)}")
                    dec = os.path.join(tmp, f"dec_{i}.wav")

                    raw_audio = (
                        w.detach().float().cpu().clamp(-1.0, 1.0).numpy()
                        .T.astype(np.float32, copy=False).tobytes()
                    )

                    cmd_src = [
                        "ffmpeg", "-y",
                        "-f", "f32le", "-ar", str(sr), "-ac", str(C),
                        "-i", "pipe:0", "-c:a", "pcm_s16le", src,
                    ]
                    subprocess.run(cmd_src, input=raw_audio,
                                   capture_output=True, check=True)

                    cmd_enc = [
                        "ffmpeg", "-y", "-i", src,
                        "-c:a", self.CODEC_ENCODERS[codec],
                        "-b:a", f"{br}k", enc,
                    ]
                    subprocess.run(cmd_enc, capture_output=True, check=True)

                    cmd_dec_wav = ["ffmpeg", "-y", "-i", enc, dec]
                    subprocess.run(cmd_dec_wav, capture_output=True, check=True)

                    cmd_pipe = [
                        "ffmpeg", "-v", "error", "-i", dec,
                        "-f", "f32le", "-acodec", "pcm_f32le",
                        "-ar", str(sr), "-ac", str(C), "pipe:1",
                    ]
                    decoded = subprocess.run(cmd_pipe, capture_output=True, check=True)
                    decoded_np = np.frombuffer(decoded.stdout, dtype=np.float32)

                    if decoded_np.size % C:
                        raise RuntimeError("decoded audio has incomplete frame")

                    w_dec = torch.from_numpy(decoded_np.reshape(-1, C).T.copy())

                    if w_dec.shape[0] != C:
                        raise RuntimeError(
                            f"channel mismatch: {w_dec.shape[0]} vs {C}")

                    if w_dec.shape[1] < T:
                        w_dec = torch.nn.functional.pad(
                            w_dec, (0, T - w_dec.shape[1]))
                    w_dec = w_dec[:, :T]
                    results.append(w_dec)

            current_wav = torch.stack(results, dim=0).to(wav.device)

        return (_make_audio(current_wav, sr),)

    @staticmethod
    def _ext(codec: str) -> str:
        return {"opus": "ogg", "aac": "m4a", "libmp3lame": "mp3"}.get(codec, "ogg")


# ──────────────────────────────────────────────
#  9. Temporal Micro-Editing (FIXED)
# ──────────────────────────────────────────────

class AudioTemporalMicroEdit:
    """
    Makes sub-millisecond cuts with proper crossfades at random positions.
    Breaks temporal coherence signatures without audible artifacts.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "edits_per_second": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1,
                }),
                "max_edit_length_ms": ("FLOAT", {
                    "default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1,
                }),
                "crossfade_ms": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 5.0, "step": 0.1,
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2**31,
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "process"
    CATEGORY = "audio/fingerprint_removal"

    def process(self, audio, edits_per_second, max_edit_length_ms,
                crossfade_ms, seed):
        wav, sr = _validate_audio(audio)
        B, C, T = wav.shape

        if seed == 0:
            seed = torch.seed()
        py_rng = random.Random(seed)

        max_edit_samples = int(max_edit_length_ms * sr / 1000)
        crossfade_samples = max(2, int(crossfade_ms * sr / 1000))
        n_edits = int(T / sr * edits_per_second)

        # Bounds guard: skip if audio too short
        min_required_len = crossfade_samples * 2 + max_edit_samples
        if T < min_required_len or n_edits == 0 or edits_per_second <= 0:
            return (audio,)

        # Precompute fade curves
        fade_in = torch.linspace(0, 1, crossfade_samples, device=wav.device)
        fade_out = torch.linspace(1, 0, crossfade_samples, device=wav.device)

        out = torch.empty_like(wav)

        for b in range(B):
            for c in range(C):
                sig = wav[b, c].clone()

                # Generate edit positions
                edit_starts = []
                for _ in range(n_edits):
                    pos = py_rng.randint(crossfade_samples,
                                         T - max_edit_samples - crossfade_samples)
                    edit_starts.append(pos)
                edit_starts.sort()

                # Apply edits from end to start to preserve indices
                for start in reversed(edit_starts):
                    length = py_rng.randint(1, max_edit_samples)
                    end = min(start + length, T - crossfade_samples)

                    before = sig[:start]
                    edited = sig[start:end]
                    after = sig[end:]

                    # Slight time-stretch on the edited region
                    shift = (py_rng.random() - 0.5) * 0.001
                    if abs(shift) > 1e-6 and len(edited) > 2:
                        new_len = max(2, int(len(edited) * (1 + shift)))
                        edited = torch.nn.functional.interpolate(
                            edited.unsqueeze(0).unsqueeze(0),
                            size=new_len, mode='linear', align_corners=False
                        ).squeeze(0).squeeze(0)

                    # Proper crossfade: overlap-add at boundaries
                    if (len(before) >= crossfade_samples
                            and len(edited) >= crossfade_samples):
                        overlap_start = (before[-crossfade_samples:] * fade_out
                                         + edited[:crossfade_samples] * fade_in)
                        sig = torch.cat([
                            before[:-crossfade_samples],
                            overlap_start,
                            edited[crossfade_samples:],
                            after
                        ])
                    elif (len(edited) >= crossfade_samples
                          and len(after) >= crossfade_samples):
                        overlap_end = (edited[-crossfade_samples:] * fade_out
                                       + after[:crossfade_samples] * fade_in)
                        sig = torch.cat([
                            before,
                            edited[:-crossfade_samples],
                            overlap_end,
                            after[crossfade_samples:]
                        ])
                    else:
                        sig = torch.cat([before, edited, after])

                    # Restore original length
                    if len(sig) > T:
                        sig = sig[:T]
                    elif len(sig) < T:
                        sig = torch.nn.functional.pad(sig, (0, T - len(sig)))

                out[b, c] = sig[:T]

        out = torch.clamp(out, -1.0, 1.0)
        return (_make_audio(out, sr),)


# ──────────────────────────────────────────────
#  10. Adaptive Parameter Selection
# ──────────────────────────────────────────────

class AudioAdaptiveParameterSelector:
    """
    Analyzes audio content and automatically selects optimal
    perturbation parameters.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_detector": (
                    ["generic", "submit_hub", "suno", "udio", "custom"],
                    {"default": "generic"}
                ),
                "transparency_priority": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
                "enable_spectral": ("BOOLEAN", {"default": True}),
                "enable_temporal": ("BOOLEAN", {"default": True}),
                "enable_codec": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "selected_params_json")
    FUNCTION = "process"
    CATEGORY = "audio/fingerprint_removal"

    def process(self, audio, target_detector, transparency_priority,
                enable_spectral, enable_temporal, enable_codec, seed):
        import json

        wav, sr = _validate_audio(audio)
        B, C, T = wav.shape

        # Analyze audio
        rms = wav.abs().mean(dim=-1).mean(dim=0)
        peak = wav.abs().max(dim=-1).values.mean(dim=0)
        crest_factor = (peak / (rms + 1e-10)).mean().item()

        n_fft = 1024
        hop = n_fft // 4
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr, device=wav.device)

        spec_mags = []
        for b in range(B):
            for c in range(C):
                S = _stft(wav[b, c].unsqueeze(0), n_fft, hop).squeeze(0)
                spec_mags.append(S.abs().mean(dim=-1))
        avg_spec = torch.stack(spec_mags).mean(dim=0)

        hf_ratio = (avg_spec[freqs > 8000].sum() / (avg_spec.sum() + 1e-10)).item()

        diff = torch.diff(wav, dim=-1)
        temporal_var = diff.abs().mean().item()

        params = self._select_parameters(
            target_detector, transparency_priority,
            crest_factor, hf_ratio, temporal_var,
            enable_spectral, enable_temporal, enable_codec
        )

        # Apply processing
        processed = wav.clone()

        if enable_spectral:
            spec_node = AudioSpectralPerturbation()
            (result,) = spec_node.process(
                {"waveform": processed, "sample_rate": sr},
                params["phase_jitter"], params["jitter_cutoff"],
                params["dither_db"], params["tilt_db"],
                params["tilt_freq"], seed
            )
            processed = result["waveform"]

        if enable_temporal and params["stretch_percent"] > 0.01:
            ts_node = AudioMicroTimeStretch()
            (result,) = ts_node.process(
                {"waveform": processed, "sample_rate": sr},
                params["stretch_percent"], seed
            )
            processed = result["waveform"]

        if enable_codec and params["codec"] != "flac":
            ce_node = AudioCodecReencode()
            (result,) = ce_node.process(
                {"waveform": processed, "sample_rate": sr},
                params["codec"], params["bitrate"]
            )
            processed = result["waveform"]

        params_json = json.dumps(params, indent=2)
        return (_make_audio(processed, sr), params_json)

    def _select_parameters(self, detector, transparency, crest, hf_ratio,
                           temporal_var, en_spec, en_temp, en_codec):
        profiles = {
            "generic": {
                "phase_jitter": 0.25, "jitter_cutoff": 8000, "dither_db": -65,
                "tilt_db": 0.5, "tilt_freq": 10000, "stretch_percent": 0.5,
                "codec": "opus", "bitrate": 320
            },
            "submit_hub": {
                "phase_jitter": 0.35, "jitter_cutoff": 7000, "dither_db": -60,
                "tilt_db": 1.0, "tilt_freq": 9000, "stretch_percent": 0.8,
                "codec": "opus", "bitrate": 256
            },
            "suno": {
                "phase_jitter": 0.30, "jitter_cutoff": 7500, "dither_db": -62,
                "tilt_db": 0.8, "tilt_freq": 9500, "stretch_percent": 0.6,
                "codec": "opus", "bitrate": 320
            },
            "udio": {
                "phase_jitter": 0.28, "jitter_cutoff": 8000, "dither_db": -63,
                "tilt_db": 0.6, "tilt_freq": 10000, "stretch_percent": 0.5,
                "codec": "aac", "bitrate": 256
            },
            "custom": {
                "phase_jitter": 0.25, "jitter_cutoff": 8000, "dither_db": -65,
                "tilt_db": 0.5, "tilt_freq": 10000, "stretch_percent": 0.5,
                "codec": "opus", "bitrate": 320
            }
        }
        p = profiles.get(detector, profiles["generic"]).copy()

        crest_adj = 1.0 - min(0.3, max(0, (crest - 6) / 20))
        hf_adj = 1.0 + min(0.4, max(0, (hf_ratio - 0.1) * 2))
        temp_adj = 1.0 - min(0.3, max(0, temporal_var * 100))

        alpha = 1.0 - transparency
        p["phase_jitter"] *= crest_adj * hf_adj * (0.5 + 0.5 * alpha)
        p["dither_db"] -= 5 * alpha
        p["tilt_db"] *= (0.5 + 0.5 * alpha)
        p["stretch_percent"] *= crest_adj * temp_adj * (0.5 + 0.5 * alpha)

        p["phase_jitter"] = max(0.05, min(0.6, p["phase_jitter"]))
        p["dither_db"] = max(-80, min(-40, p["dither_db"]))
        p["tilt_db"] = max(-1.5, min(1.5, p["tilt_db"]))
        p["stretch_percent"] = max(0.1, min(1.5, p["stretch_percent"]))

        if not en_spec:
            p["phase_jitter"] = 0
            p["dither_db"] = -90
            p["tilt_db"] = 0
        if not en_temp:
            p["stretch_percent"] = 0
        if not en_codec:
            p["codec"] = "flac"
            p["bitrate"] = 1411

        return p


# ──────────────────────────────────────────────
#  11. Spectral Envelope Warping (FIXED)
# ──────────────────────────────────────────────

class AudioSpectralEnvelopeWarping:
    """
    Non-linear frequency axis warping that preserves perceptual
    timbre but breaks statistical signatures.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "warp_strength": ("FLOAT", {
                    "default": 0.03, "min": 0.0, "max": 0.1, "step": 0.005,
                }),
                "warp_smoothness": ("FLOAT", {
                    "default": 4.0, "min": 1.0, "max": 10.0, "step": 0.5,
                }),
                "preserve_formants": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2**31,
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "process"
    CATEGORY = "audio/fingerprint_removal"

    def process(self, audio, warp_strength, warp_smoothness,
                preserve_formants, seed):
        wav, sr = _validate_audio(audio)
        B, C, T = wav.shape

        if warp_strength <= 0:
            return (audio,)

        if seed == 0:
            seed = torch.seed()
        gen = torch.Generator(device=wav.device).manual_seed(seed)

        n_fft = 2048
        hop = n_fft // 4
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr, device=wav.device)
        erb_freqs = _hz_to_erb(freqs)
        F = len(freqs)

        out = torch.empty_like(wav)

        for b in range(B):
            for c in range(C):
                sig = wav[b, c]
                S = _stft(sig.unsqueeze(0), n_fft, hop).squeeze(0)
                mag = S.abs()
                phase = S.angle()

                # Generate smooth warp offsets in ERB domain
                n_control = 32
                erb_min, erb_max = erb_freqs[0].item(), erb_freqs[-1].item()
                control_erbs = torch.linspace(erb_min, erb_max, n_control,
                                              device=wav.device)

                warp_offsets = (
                    torch.rand(n_control, device=wav.device, generator=gen) * 2 - 1
                ) * warp_strength

                # Smooth
                kernel_size = max(1, int(warp_smoothness * 2))
                if kernel_size > 1:
                    warp_offsets = torch.nn.functional.avg_pool1d(
                        warp_offsets.unsqueeze(0).unsqueeze(0),
                        kernel_size=kernel_size, stride=1,
                        padding=kernel_size // 2
                    ).squeeze(0).squeeze(0)

                # Protect formant regions
                if preserve_formants:
                    formant_mask = (freqs >= 500) & (freqs <= 4000)
                    formant_erb = erb_freqs[formant_mask]
                    if len(formant_erb) > 0:
                        for i, erb_val in enumerate(control_erbs):
                            dist = torch.abs(formant_erb - erb_val).min().item()
                            taper = min(1.0, dist / _hz_to_erb(
                                torch.tensor(1000.0)).item())
                            warp_offsets[i] *= taper

                # Interpolate warp to all frequency bins
                warp_interp = torch.nn.functional.interpolate(
                    warp_offsets.unsqueeze(0).unsqueeze(0),
                    size=F, mode='linear', align_corners=False
                ).squeeze(0).squeeze(0)

                # Compute warped frequency positions
                warped_erb = erb_freqs + warp_interp
                warped_erb = torch.clamp(warped_erb, erb_min, erb_max)
                warped_hz = _erb_to_hz(warped_erb)

                # Gather from warped positions using linear interpolation indices
                freq_range = freqs[-1] - freqs[0]
                if freq_range < 1.0:
                    out[b, c] = sig
                    continue

                norm_idx = ((warped_hz - freqs[0]) / freq_range
                            * (F - 1)).clamp(0, F - 1)
                idx_low = norm_idx.long().clamp(max=F - 2)
                idx_high = idx_low + 1
                frac = (norm_idx - idx_low.float()).unsqueeze(-1)  # (F, 1)

                mag_warped = mag[idx_low] * (1 - frac) + mag[idx_high] * frac
                phase_warped = phase[idx_low] * (1 - frac) + phase[idx_high] * frac

                S_out = mag_warped * torch.exp(1j * phase_warped)
                sig_out = _istft(S_out.unsqueeze(0), n_fft, hop, T).squeeze(0)
                out[b, c] = sig_out

        out = torch.clamp(out, -1.0, 1.0)
        return (_make_audio(out, sr),)


# ──────────────────────────────────────────────
#  12. Enhanced Full Pipeline (v2)
# ──────────────────────────────────────────────

class AudioFingerprintPipelineV2:
    """
    Full pipeline: Spectral Perturbation -> Advanced Phase Scrambling
    -> Mid/Side -> Psychoacoustic Noise -> Temporal Edit -> Warping
    -> Time-Stretch -> Multi-Codec Chain.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "phase_jitter_strength": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01
                }),
                "jitter_cutoff_hz": ("FLOAT", {
                    "default": 8000.0, "min": 2000.0, "max": 20000.0, "step": 100.0
                }),
                "dither_db": ("FLOAT", {
                    "default": -65.0, "min": -90.0, "max": -40.0, "step": 1.0
                }),
                "tilt_db": ("FLOAT", {
                    "default": 0.0, "min": -3.0, "max": 3.0, "step": 0.1
                }),
                "enable_adv_phase": ("BOOLEAN", {"default": True}),
                "adv_phase_strength": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 0.6, "step": 0.01
                }),
                "enable_mid_side": ("BOOLEAN", {"default": True}),
                "mid_jitter": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 0.5, "step": 0.01
                }),
                "side_jitter": ("FLOAT", {
                    "default": 0.35, "min": 0.0, "max": 0.7, "step": 0.01
                }),
                "enable_psychoacoustic": ("BOOLEAN", {"default": True}),
                "target_snr_db": ("FLOAT", {
                    "default": 40.0, "min": 20.0, "max": 80.0, "step": 1.0
                }),
                "enable_temporal_edit": ("BOOLEAN", {"default": True}),
                "edits_per_second": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 3.0, "step": 0.1
                }),
                "enable_warping": ("BOOLEAN", {"default": True}),
                "warp_strength": ("FLOAT", {
                    "default": 0.02, "min": 0.0, "max": 0.08, "step": 0.005
                }),
                "stretch_percent": ("FLOAT", {
                    "default": 0.5, "min": -2.0, "max": 2.0, "step": 0.1
                }),
                "enable_codec_chain": ("BOOLEAN", {"default": True}),
                "codec_chain": (
                    ["opus_to_aac", "opus_to_mp3", "aac_to_mp3", "opus_to_aac_to_mp3"],
                    {"default": "opus_to_aac"}
                ),
                "bitrate_kbps": ("INT", {
                    "default": 256, "min": 96, "max": 512, "step": 32
                }),
                "final_bitrate_kbps": ("INT", {
                    "default": 320, "min": 128, "max": 512, "step": 32
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "process"
    CATEGORY = "audio/fingerprint_removal"

    def process(self, audio, phase_jitter_strength, jitter_cutoff_hz,
                dither_db, tilt_db,
                enable_adv_phase, adv_phase_strength,
                enable_mid_side, mid_jitter, side_jitter,
                enable_psychoacoustic, target_snr_db,
                enable_temporal_edit, edits_per_second,
                enable_warping, warp_strength,
                stretch_percent,
                enable_codec_chain, codec_chain, bitrate_kbps,
                final_bitrate_kbps, seed):

        # 1 - Spectral Perturbation
        spec_node = AudioSpectralPerturbation()
        (audio,) = spec_node.process(
            audio, phase_jitter_strength, jitter_cutoff_hz,
            dither_db, tilt_db, 10000.0, seed
        )

        # 2 - Advanced Phase Scrambling
        if enable_adv_phase:
            adv_node = AudioAdvancedPhaseScrambling()
            (audio,) = adv_node.process(
                audio, adv_phase_strength, 0.7, 1.5, 2000.0, seed + 1
            )

        # 3 - Mid/Side Perturbation
        if enable_mid_side:
            ms_node = AudioMidSidePerturbation()
            (audio,) = ms_node.process(
                audio, mid_jitter, side_jitter, -70.0, -60.0, 0.05, seed + 2
            )

        # 4 - Psychoacoustic Noise Shaping
        if enable_psychoacoustic:
            pa_node = AudioPsychoacousticNoiseShaping()
            (audio,) = pa_node.process(
                audio, target_snr_db, 6.0, -90.0, seed + 3
            )

        # 5 - Temporal Micro-Editing
        if enable_temporal_edit:
            te_node = AudioTemporalMicroEdit()
            (audio,) = te_node.process(
                audio, edits_per_second, 2.0, 0.5, seed + 4
            )

        # 6 - Spectral Envelope Warping
        if enable_warping:
            warp_node = AudioSpectralEnvelopeWarping()
            (audio,) = warp_node.process(
                audio, warp_strength, 4.0, True, seed + 5
            )

        # 7 - Micro Time-Stretch
        if abs(stretch_percent) > 0.01:
            ts_node = AudioMicroTimeStretch()
            (audio,) = ts_node.process(audio, stretch_percent, seed + 6)

        # 8 - Multi-Codec Chain
        if enable_codec_chain:
            mc_node = AudioMultiCodecChain()
            (audio,) = mc_node.process(
                audio, codec_chain, bitrate_kbps, final_bitrate_kbps
            )

        return (audio,)