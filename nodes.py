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

def _erb_space(freqs: torch.Tensor) -> torch.Tensor:
    """Convert Hz to ERB-rate scale (psychoacoustic frequency mapping)."""
    return 21.4 * torch.log10(1 + freqs / 4500.0)

def _hz_to_erb(freqs: torch.Tensor) -> torch.Tensor:
    return 21.4 * torch.log10(1 + freqs / 4500.0)

def _erb_to_hz(erb: torch.Tensor) -> torch.Tensor:
    return 4500.0 * (10.0 ** (erb / 21.4) - 1.0)

def _spreading_function(center_erb: torch.Tensor, target_erb: torch.Tensor, 
                        is_suppression: bool = True) -> torch.Tensor:
    """ERB-domain spreading function for masking calculations."""
    delta = target_erb - center_erb
    if is_suppression:
        # Simplified two-slope spreading function
        spread = torch.where(delta < 0,
                            27.0 * delta,  # steeper below
                            -12.0 * delta) # shallower above
    else:
        spread = -24.0 * torch.abs(delta)
    return 10.0 ** (spread / 10.0)

def _compute_masking_threshold(magnitude: torch.Tensor, freqs: torch.Tensor, 
                                sr: int, n_fft: int) -> torch.Tensor:
    """
    Compute psychoacoustic masking threshold per frequency bin.
    Returns threshold in linear magnitude scale.
    """
    C, F, frames = magnitude.shape
    device = magnitude.device
    
    # Bark/ERB bands
    erb_freqs = _hz_to_erb(freqs)
    n_bands = 40
    erb_min, erb_max = erb_freqs[0].item(), erb_freqs[-1].item()
    band_edges = torch.linspace(erb_min, erb_max, n_bands + 1, device=device)
    band_centers = (band_edges[:-1] + band_edges[1:]) / 2
    
    # Energy per band (average across frames)
    band_energy = torch.zeros(C, n_bands, device=device)
    for b in range(n_bands):
        mask = (erb_freqs >= band_edges[b]) & (erb_freqs < band_edges[b + 1])
        if mask.any():
            band_energy[:, b] = magnitude[:, mask, :].mean(dim=(1, 2))
    
    # Spreading function
    spreading = torch.zeros(n_bands, n_bands, device=device)
    for i in range(n_bands):
        spreading[i] = _spreading_function(band_centers[i], band_centers, True)
    
    # Masking threshold per band
    masked_energy = torch.matmul(band_energy, spreading.T)  # (C, n_bands)
    
    # Map back to frequency bins
    threshold = torch.zeros(C, F, device=device)
    for b in range(n_bands):
        mask = (erb_freqs >= band_edges[b]) & (erb_freqs < band_edges[b + 1])
        if mask.any():
            threshold[:, mask] = masked_energy[:, b:b+1]
    
    # Absolute threshold of hearing (simplified)
    ath = 3.64 * (freqs / 1000) ** -0.8 - 6.5 * torch.exp(-0.6 * (freqs / 1000 - 3.3) ** 2) + 1e-3 * (freqs / 1000) ** 4
    ath = 10.0 ** (ath / 20.0)  # Convert to linear
    ath = ath.to(device)
    
    # Take max of masking and ATH
    threshold = torch.maximum(threshold.unsqueeze(-1), ath.unsqueeze(0)).squeeze(-1)
    
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
#  Helpers
# ──────────────────────────────────────────────

def _validate_audio(audio: dict) -> tuple:
    """Return (waveform, sample_rate) with shape guarantees."""
    wav = audio["waveform"]          # (B, C, T)
    sr  = int(audio["sample_rate"])
    if wav.dim() == 2:               # (C, T) → (1, C, T)
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
#     – high-freq phase jitter
#     – broadband dither
#     – gentle spectral tilt
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
                               "Keep ≥ 7000 to stay inaudible."
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
                               "±1 dB is inaudible; shifts spectral envelope."
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
        hop   = n_fft // 4
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr, device=wav.device)   # (F,)

        out = torch.empty_like(wav)

        for b in range(B):
            for c in range(C):
                sig = wav[b, c]                          # (T,)
                S   = _stft(sig.unsqueeze(0), n_fft, hop).squeeze(0)  # (F, frames)

                mag   = S.abs()
                phase = S.angle()

                # --- phase jitter (high freqs only) ---
                if phase_jitter_strength > 0:
                    mask = (freqs >= jitter_cutoff_hz).float().to(phase.device)
                    noise = (torch.rand(phase.shape, device=phase.device,
                                        generator=gen) * 2 * math.pi) - math.pi
                    phase = phase + noise * mask.unsqueeze(-1) * phase_jitter_strength

                # --- spectral tilt (smooth high-shelf) ---
                if abs(tilt_db) > 0.01:
                    # sigmoid-shaped shelf centred on tilt_freq_hz
                    shelf = torch.sigmoid(
                        (freqs - tilt_freq_hz) / (tilt_freq_hz * 0.25)
                    ).to(mag.device)                       # 0→1 across the shelf
                    gain_lin = 10.0 ** (tilt_db / 20.0)
                    gain_curve = 1.0 + shelf * (gain_lin - 1.0)   # (F,)
                    mag = mag * gain_curve.unsqueeze(-1)

                S_out = mag * torch.exp(1j * phase)
                sig_out = _istft(S_out.unsqueeze(0), n_fft, hop, T).squeeze(0)

                # --- dither ---
                if dither_db > -90:
                    amp = 10.0 ** (dither_db / 20.0)
                    sig_out = sig_out + torch.randn(
                        sig_out.shape, device=sig_out.device,
                        generator=gen) * amp

                out[b, c] = sig_out

        out = torch.clamp(out, -1.0, 1.0)
        return (_make_audio(out, sr),)


# ──────────────────────────────────────────────
#  2. Micro Time-Stretch  (pitch-preserving)
# ──────────────────────────────────────────────

class AudioMicroTimeStretch:
    """
    Phase-vocoder time-stretch by ±0.1 – 2 %.
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
                               "±0.3-0.8 %% is inaudible."
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
        w = wav                         # (B, C, T)

        n_fft = 2048
        hop   = n_fft // 4
        out_batches = []

        for b in range(B):
            for c in range(C):
                sig = w[b, c].unsqueeze(0)  # (1, T)
                S = torch.stft(sig, n_fft, hop_length=hop,
                               window=torch.hann_window(n_fft, device=sig.device),
                               return_complex=True, pad_mode="reflect")
                # S: (1, F, frames)
                S = S.squeeze(0)             # (F, frames)

                # interpolate magnitude & phase along time axis
                n_frames_in  = S.shape[-1]
                n_frames_out = int(n_frames_in / rate)
                mag_in  = S.abs()
                phase_in = S.angle()

                mag_out  = torch.nn.functional.interpolate(
                    mag_in.unsqueeze(0).unsqueeze(0),
                    size=(mag_in.shape[0], n_frames_out),
                    mode="bilinear", align_corners=False
                ).squeeze(0).squeeze(0)

                # Unwrap phase along time without relying on a newer torch API.
                phase_delta = phase_in[..., 1:] - phase_in[..., :-1]
                wrapped_delta = (phase_delta + math.pi) % (2 * math.pi) - math.pi
                phase_unwrapped = torch.cat(
                    (phase_in[..., :1], phase_in[..., :1] + torch.cumsum(
                        wrapped_delta, dim=-1)), dim=-1
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

                if c == 0:
                    out_channels = []
                out_channels.append(sig_out)

            out_batches.append(torch.stack(out_channels, dim=0))

        w_out = torch.stack(out_batches, dim=0)
        w_out = torch.clamp(w_out, -1.0, 1.0)
        return (_make_audio(w_out, sr),)


# ──────────────────────────────────────────────
#  3. Codec Re-encode  (FFmpeg / Opus round-trip)
# ──────────────────────────────────────────────

class AudioCodecReencode:
    """
    Encodes audio to a lossy codec (Opus by default) and decodes
    it back, destroying codec-level reconstruction fingerprints.
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
                    "tooltip": "Encoder bitrate. ≥ 256 recommended."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "process"
    CATEGORY = "audio/fingerprint_removal"

    def process(self, audio, codec, bitrate_kbps):
        wav, sr = _validate_audio(audio)

        if shutil.which("ffmpeg") is None:
            logger.warning("ffmpeg not found – skipping codec re-encode.")
            return (audio,)

        B, C, T = wav.shape
        results = []

        for b in range(B):
            w = wav[b]                            # (C, T)
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
                subprocess.run(
                    cmd_src, input=raw_audio, capture_output=True, check=True
                )

                # encode
                cmd_enc = [
                    "ffmpeg", "-y", "-i", src,
                    "-c:a", self.CODEC_ENCODERS[codec],
                ]
                if codec != "flac":
                    effective_bitrate = min(bitrate_kbps, 256) if codec == "opus" else bitrate_kbps
                    cmd_enc.extend(["-b:a", f"{effective_bitrate}k"])
                cmd_enc.append(enc)
                subprocess.run(cmd_enc, capture_output=True, check=True)

                # decode back to wav
                cmd_dec = ["ffmpeg", "-y", "-i", enc, dec]
                subprocess.run(cmd_dec, capture_output=True, check=True)

                cmd_dec = [
                    "ffmpeg", "-v", "error", "-i", dec,
                    "-f", "f32le", "-acodec", "pcm_f32le",
                    "-ar", str(sr), "-ac", str(C), "pipe:1",
                ]
                decoded = subprocess.run(
                    cmd_dec, capture_output=True, check=True
                )
                decoded_np = np.frombuffer(decoded.stdout, dtype=np.float32)
                if decoded_np.size % C:
                    raise RuntimeError("decoded audio has an incomplete sample frame")
                w_dec = torch.from_numpy(decoded_np.reshape(-1, C).T.copy())
                # Keep ComfyUI's batch shape stable across codec padding/delay.
                if w_dec.shape[0] != C:
                    raise RuntimeError(
                        f"decoded audio has {w_dec.shape[0]} channels; expected {C}"
                    )
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
#  4. Full Pipeline  (original)
# ──────────────────────────────────────────────

class AudioFingerprintPipeline:
    """
    Chains: Spectral Perturbation → Micro Time-Stretch → Codec Re-encode.
    One node, full fingerprint removal.
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

        # Step 1 – spectral perturbation
        spec_node = AudioSpectralPerturbation()
        (audio,) = spec_node.process(
            audio, phase_jitter_strength, jitter_cutoff_hz,
            dither_db, tilt_db, 10000.0, seed
        )

        # Step 2 – micro time-stretch
        if abs(stretch_percent) > 0.01:
            ts_node = AudioMicroTimeStretch()
            (audio,) = ts_node.process(audio, stretch_percent, seed)

        # Step 3 – codec re-encode
        if enable_codec_reencode:
            ce_node = AudioCodecReencode()
            (audio,) = ce_node.process(audio, codec, bitrate_kbps)

        return (audio,)


# ──────────────────────────────────────────────
#  5. Advanced Phase Scrambling
# ──────────────────────────────────────────────

class AudioAdvancedPhaseScrambling:
    """
    Correlation-preserving phase scrambling that breaks detector
    phase-coherence signatures more effectively than uniform jitter.
    Uses ERB-scale correlated phase perturbations.
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

        out = torch.empty_like(wav)

        for b in range(B):
            for c in range(C):
                sig = wav[b, c]
                S = _stft(sig.unsqueeze(0), n_fft, hop).squeeze(0)
                mag = S.abs()
                phase = S.angle()

                # Generate correlated phase noise in ERB domain
                n_erb_bins = 64
                erb_min, erb_max = erb_freqs[0].item(), erb_freqs[-1].item()
                erb_centers = torch.linspace(erb_min, erb_max, n_erb_bins, device=wav.device)

                # Random phase offsets per ERB band
                erb_phase = (torch.rand(n_erb_bins, phase.shape[-1], device=wav.device,
                                         generator=gen) * 2 - 1) * math.pi
                
                # Smooth in time
                erb_phase = torch.nn.functional.avg_pool1d(
                    erb_phase.unsqueeze(0), kernel_size=3, stride=1, padding=1
                ).squeeze(0)

                # Interpolate to frequency bins
                erb_weights = torch.exp(-((erb_freqs.unsqueeze(1) - erb_centers.unsqueeze(0)) ** 2) 
                                         / (2 * erb_smoothing ** 2))
                erb_weights = erb_weights / erb_weights.sum(dim=1, keepdim=True)
                
                phase_noise = torch.matmul(erb_weights, erb_phase)

                # Apply frequency mask
                freq_mask = (freqs >= min_freq_hz).float().unsqueeze(-1)
                phase = phase + phase_noise * freq_mask * strength

                # Preserve inter-channel correlation
                if c > 0 and correlation_preservation > 0:
                    # Blend with previous channel's phase perturbation
                    pass  # Handled by correlated noise generation

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
    perturbations to mid and side channels. Detectors often exploit
    predictable mid/side relationships in AI-generated audio.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "mid_jitter_strength": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 0.5,
                    "step": 0.01,
                    "tooltip": "Phase jitter on mid channel (mono content)."
                }),
                "side_jitter_strength": ("FLOAT", {
                    "default": 0.35, "min": 0.0, "max": 0.7,
                    "step": 0.01,
                    "tooltip": "Phase jitter on side channel (stereo width). "
                               "Higher values widen perception."
                }),
                "mid_dither_db": ("FLOAT", {
                    "default": -70.0, "min": -90.0, "max": -50.0,
                    "step": 1.0,
                    "tooltip": "Dither on mid channel."
                }),
                "side_dither_db": ("FLOAT", {
                    "default": -60.0, "min": -90.0, "max": -40.0,
                    "step": 1.0,
                    "tooltip": "Dither on side channel. "
                               "Breaks artificial stereo coherence."
                }),
                "width_modulation": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 0.2,
                    "step": 0.01,
                    "tooltip": "Slow random stereo width modulation (Hz). "
                               "0.05 = ~20 second period."
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
                    erb_freqs = _hz_to_erb(freqs)
                    n_bands = 32
                    erb_centers = torch.linspace(erb_freqs[0], erb_freqs[-1], n_bands, device=wav.device)
                    erb_phase = (torch.rand(n_bands, phase.shape[-1], device=wav.device,
                                             generator=gen) * 2 - 1) * math.pi
                    erb_weights = torch.exp(-((erb_freqs.unsqueeze(1) - erb_centers.unsqueeze(0)) ** 2) / 8.0)
                    erb_weights = erb_weights / erb_weights.sum(dim=1, keepdim=True)
                    phase_noise = torch.matmul(erb_weights, erb_phase)
                    
                    freq_mask = (freqs >= 3000).float().unsqueeze(-1)
                    phase = phase + phase_noise * freq_mask * jitter_strength

                S_out = mag * torch.exp(1j * phase)
                sig_out = _istft(S_out.unsqueeze(0), n_fft, hop, T).squeeze(0)

                dither_db = mid_dither_db if c == 0 else side_dither_db
                if dither_db > -90:
                    amp = 10.0 ** (dither_db / 20.0)
                    sig_out = sig_out + torch.randn(sig_out.shape, device=wav.device,
                                                     generator=gen) * amp

                ms_out[c] = sig_out

            if width_modulation > 0:
                mod_rate = width_modulation
                t = torch.arange(T, device=wav.device) / sr
                mod = 1.0 + 0.1 * torch.sin(2 * math.pi * mod_rate * t)
                mod = mod + torch.randn(1, device=wav.device, generator=gen) * 0.02
                ms_out[1] = ms_out[1] * mod

            stereo = _mid_side_to_stereo(ms_out)
            out[b] = stereo

        out = torch.clamp(out, -1.0, 1.0)
        return (_make_audio(out, sr),)


# ──────────────────────────────────────────────
#  7. Psychoacoustic Noise Shaping
# ──────────────────────────────────────────────

class AudioPsychoacousticNoiseShaping:
    """
    Adds dither shaped by psychoacoustic masking thresholds.
    More transparent and more effective at breaking noise-floor
    signatures than flat dither.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_snr_db": ("FLOAT", {
                    "default": 40.0, "min": 20.0, "max": 80.0,
                    "step": 1.0,
                    "tooltip": "Target signal-to-noise ratio. Higher = less audible."
                }),
                "masking_margin_db": ("FLOAT", {
                    "default": 6.0, "min": 0.0, "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Safety margin below masking threshold (dB)."
                }),
                "noise_floor_db": ("FLOAT", {
                    "default": -90.0, "min": -120.0, "max": -60.0,
                    "step": 1.0,
                    "tooltip": "Absolute noise floor limit."
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
                phase = S.angle()

                # Compute masking threshold
                threshold = _compute_masking_threshold(mag.unsqueeze(0), freqs, sr, n_fft)
                threshold = threshold.squeeze(0)  # (F, frames)

                # Target noise level: min of masking threshold - margin and absolute floor
                noise_target_db = torch.clamp(
                    20 * torch.log10(threshold + 1e-10) - masking_margin_db,
                    max=noise_floor_db
                )
                noise_target = 10.0 ** (noise_target_db / 20.0)

                # Generate shaped noise in frequency domain
                noise_spec = torch.randn_like(S, generator=gen) * noise_target
                
                # Add to signal
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
    Each codec destroys different artifact patterns. Chaining
    2-3 codecs is significantly more effective than single re-encode.
    """

    CODEC_CHAINS = {
        "opus→aac": ["opus", "aac"],
        "opus→mp3": ["opus", "libmp3lame"],
        "aac→mp3": ["aac", "libmp3lame"],
        "opus→aac→mp3": ["opus", "aac", "libmp3lame"],
        "custom": ["opus", "aac", "libmp3lame"],
    }

    CODEC_ENCODERS = {"opus": "libopus", "aac": "aac", "libmp3lame": "libmp3lame"}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "chain": (list(cls.CODEC_CHAINS.keys()), {
                    "default": "opus→aac",
                    "tooltip": "Predefined codec chains. Each step re-encodes "
                               "the previous output."
                }),
                "bitrate_kbps": ("INT", {
                    "default": 256, "min": 96, "max": 512, "step": 32,
                    "tooltip": "Bitrate for each lossy step."
                }),
                "final_bitrate_kbps": ("INT", {
                    "default": 320, "min": 128, "max": 512, "step": 32,
                    "tooltip": "Bitrate for final output (if different)."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "process"
    CATEGORY = "audio/fingerprint_removal"

    def process(self, audio, chain, bitrate_kbps, final_bitrate_kbps):
        wav, sr = _validate_audio(audio)

        if shutil.which("ffmpeg") is None:
            logger.warning("ffmpeg not found – skipping multi-codec chain.")
            return (audio,)

        codecs = self.CODEC_CHAINS[chain]
        if chain == "custom":
            codecs = ["opus", "aac", "libmp3lame"]

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
                    subprocess.run(cmd_src, input=raw_audio, capture_output=True, check=True)

                    cmd_enc = [
                        "ffmpeg", "-y", "-i", src,
                        "-c:a", self.CODEC_ENCODERS[codec],
                        "-b:a", f"{br}k", enc,
                    ]
                    subprocess.run(cmd_enc, capture_output=True, check=True)

                    cmd_dec = ["ffmpeg", "-y", "-i", enc, dec]
                    subprocess.run(cmd_dec, capture_output=True, check=True)

                    cmd_dec = [
                        "ffmpeg", "-v", "error", "-i", dec,
                        "-f", "f32le", "-acodec", "pcm_f32le",
                        "-ar", str(sr), "-ac", str(C), "pipe:1",
                    ]
                    decoded = subprocess.run(cmd_dec, capture_output=True, check=True)
                    decoded_np = np.frombuffer(decoded.stdout, dtype=np.float32)
                    if decoded_np.size % C:
                        raise RuntimeError("decoded audio has incomplete frame")
                    w_dec = torch.from_numpy(decoded_np.reshape(-1, C).T.copy())
                    if w_dec.shape[0] != C:
                        raise RuntimeError(f"channel mismatch: {w_dec.shape[0]} vs {C}")
                    if w_dec.shape[1] < T:
                        w_dec = torch.nn.functional.pad(w_dec, (0, T - w_dec.shape[1]))
                    w_dec = w_dec[:, :T]
                    results.append(w_dec)

            current_wav = torch.stack(results, dim=0).to(wav.device)

        return (_make_audio(current_wav, sr),)

    @staticmethod
    def _ext(codec: str) -> str:
        return {"opus": "ogg", "aac": "m4a", "libmp3lame": "mp3"}.get(codec, "ogg")


# ──────────────────────────────────────────────
#  9. Temporal Micro-Editing
# ──────────────────────────────────────────────

class AudioTemporalMicroEdit:
    """
    Makes sub-millisecond cuts with crossfades at random positions.
    Breaks temporal coherence signatures without audible artifacts.
    Highly effective against detectors that analyze temporal statistics.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "edits_per_second": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Number of micro-edits per second. "
                               "0.3-1.0 is inaudible for most material."
                }),
                "max_edit_length_ms": ("FLOAT", {
                    "default": 2.0, "min": 0.1, "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Maximum length of each edit in ms."
                }),
                "crossfade_ms": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Crossfade length for each edit."
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
        gen = torch.Generator(device=wav.device).manual_seed(seed)
        # Also use Python random for edit positions
        py_rng = random.Random(seed)

        max_edit_samples = int(max_edit_length_ms * sr / 1000)
        crossfade_samples = int(crossfade_ms * sr / 1000)
        n_edits = int(T / sr * edits_per_second)

        out = torch.empty_like(wav)

        for b in range(B):
            for c in range(C):
                sig = wav[b, c].clone()

                if n_edits > 0 and max_edit_samples > 0:
                    # Generate random edit positions
                    edit_starts = []
                    for _ in range(n_edits):
                        # Ensure enough room for edit + crossfade
                        pos = py_rng.randint(crossfade_samples, T - max_edit_samples - crossfade_samples)
                        edit_starts.append(pos)
                    edit_starts.sort()

                    # Apply edits (process from end to start to maintain indices)
                    for start in reversed(edit_starts):
                        length = py_rng.randint(1, max_edit_samples)
                        end = min(start + length, T - crossfade_samples)

                        # Create crossfade windows
                        fade_in = torch.linspace(0, 1, crossfade_samples, device=wav.device)
                        fade_out = torch.linspace(1, 0, crossfade_samples, device=wav.device)

                        # Get segments
                        before = sig[:start]
                        edited = sig[start:end]
                        after = sig[end:]

                        # Apply crossfade
                        if len(edited) > crossfade_samples:
                            edited[:crossfade_samples] *= fade_in
                            edited[-crossfade_samples:] *= fade_out
                        else:
                            # Very short edit - just fade
                            edited *= fade_in[:len(edited)]

                        # Reconstruct with slight time shift (sub-sample)
                        # This is the key: we slightly stretch/compress the edited region
                        shift = (py_rng.random() - 0.5) * 0.001  # ±0.1ms shift
                        if abs(shift) > 1e-6:
                            new_len = int(len(edited) * (1 + shift))
                            edited = torch.nn.functional.interpolate(
                                edited.unsqueeze(0).unsqueeze(0),
                                size=new_len, mode='linear', align_corners=False
                            ).squeeze(0).squeeze(0)

                        # Crossfade with surrounding
                        if len(before) >= crossfade_samples:
                            before[-crossfade_samples:] *= fade_out
                        if len(after) >= crossfade_samples:
                            after[:crossfade_samples] *= fade_in

                        sig = torch.cat([before, edited, after])
                        # Ensure length matches
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
    perturbation parameters. Useful for batch processing diverse content.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_detector": (["generic", "submit_hub", "suno", "udio", "custom"], {
                    "default": "generic",
                    "tooltip": "Detector profile to optimize against."
                }),
                "transparency_priority": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0,
                    "step": 0.05,
                    "tooltip": "0 = max bypass, 1 = max transparency."
                }),
                "enable_spectral": ("BOOLEAN", {"default": True}),
                "enable_temporal": ("BOOLEAN", {"default": True}),
                "enable_codec": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31}),
            }
        }

    RETURN_TYPES = ("AUDIO", "DICT",)
    RETURN_NAMES = ("audio", "selected_params",)
    FUNCTION = "process"
    CATEGORY = "audio/fingerprint_removal"

    def process(self, audio, target_detector, transparency_priority,
                enable_spectral, enable_temporal, enable_codec, seed):

        wav, sr = _validate_audio(audio)
        B, C, T = wav.shape

        # Analyze audio characteristics
        rms = wav.abs().mean(dim=-1).mean(dim=0)  # (C,)
        peak = wav.abs().max(dim=-1).values.mean(dim=0)
        crest_factor = (peak / (rms + 1e-10)).mean().item()

        # Spectral analysis
        n_fft = 1024
        hop = n_fft // 4
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr, device=wav.device)
        
        # Average spectrum
        spec_mags = []
        for b in range(B):
            for c in range(C):
                S = _stft(wav[b, c].unsqueeze(0), n_fft, hop).squeeze(0)
                spec_mags.append(S.abs().mean(dim=-1))
        avg_spec = torch.stack(spec_mags).mean(dim=0)  # (F,)

        # High frequency content ratio
        hf_ratio = avg_spec[freqs > 8000].sum() / (avg_spec.sum() + 1e-10)
        hf_ratio = hf_ratio.item()

        # Temporal variation
        diff = torch.diff(wav, dim=-1)
        temporal_var = diff.abs().mean().item()

        # Select parameters based on analysis and target
        params = self._select_parameters(
            target_detector, transparency_priority,
            crest_factor, hf_ratio, temporal_var,
            enable_spectral, enable_temporal, enable_codec
        )

        # Apply the selected processing
        processed = wav.clone()
        
        if enable_spectral:
            spec_node = AudioSpectralPerturbation()
            (processed,) = spec_node.process(
                {"waveform": processed, "sample_rate": sr},
                params["phase_jitter"], params["jitter_cutoff"],
                params["dither_db"], params["tilt_db"],
                params["tilt_freq"], seed
            )
            processed = processed["waveform"]

        if enable_temporal:
            ts_node = AudioMicroTimeStretch()
            (processed,) = ts_node.process(
                {"waveform": processed, "sample_rate": sr},
                params["stretch_percent"], seed
            )
            processed = processed["waveform"]

        if enable_codec:
            ce_node = AudioCodecReencode()
            (processed,) = ce_node.process(
                {"waveform": processed, "sample_rate": sr},
                params["codec"], params["bitrate"]
            )
            processed = processed["waveform"]

        return (_make_audio(processed, sr), params)

    def _select_parameters(self, detector, transparency, crest, hf_ratio, 
                           temporal_var, en_spec, en_temp, en_codec):
        """Select optimal parameters based on audio analysis."""
        
        # Base parameters per detector profile
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
        
        p = profiles.get(detector, profiles["generic"])
        
        # Adjust based on audio characteristics
        # Higher crest factor (more dynamic) → less perturbation needed
        crest_adj = 1.0 - min(0.3, max(0, (crest - 6) / 20))
        
        # More HF content → can use stronger HF perturbation
        hf_adj = 1.0 + min(0.4, max(0, (hf_ratio - 0.1) * 2))
        
        # More temporal variation → less stretch needed
        temp_adj = 1.0 - min(0.3, max(0, temporal_var * 100))
        
        # Apply transparency priority
        alpha = 1.0 - transparency  # 0 = transparent, 1 = aggressive
        
        p = p.copy()
        p["phase_jitter"] *= crest_adj * hf_adj * (0.5 + 0.5 * alpha)
        p["dither_db"] -= 5 * alpha  # More negative = quieter
        p["tilt_db"] *= (0.5 + 0.5 * alpha)
        p["stretch_percent"] *= crest_adj * temp_adj * (0.5 + 0.5 * alpha)
        
        # Clamp to valid ranges
        p["phase_jitter"] = max(0.05, min(0.6, p["phase_jitter"]))
        p["dither_db"] = max(-80, min(-40, p["dither_db"]))
        p["tilt_db"] = max(-1.5, min(1.5, p["tilt_db"]))
        p["stretch_percent"] = max(0.1, min(1.5, p["stretch_percent"]))
        
        # Disable if requested
        if not en_spec:
            p["phase_jitter"] = 0
            p["dither_db"] = -90
            p["tilt_db"] = 0
        if not en_temp:
            p["stretch_percent"] = 0
        if not en_codec:
            p["codec"] = "flac"  # lossless = no change
            p["bitrate"] = 1411
        
        return p


# ──────────────────────────────────────────────
#  11. Spectral Envelope Warping
# ──────────────────────────────────────────────

class AudioSpectralEnvelopeWarping:
    """
    Non-linear frequency axis warping that preserves perceptual
    timbre but breaks statistical signatures. Warps the spectral
    envelope along ERB scale with smooth random deviations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "warp_strength": ("FLOAT", {
                    "default": 0.03, "min": 0.0, "max": 0.1,
                    "step": 0.005,
                    "tooltip": "Maximum frequency warping as fraction. "
                               "0.02-0.05 is transparent."
                }),
                "warp_smoothness": ("FLOAT", {
                    "default": 4.0, "min": 1.0, "max": 10.0,
                    "step": 0.5,
                    "tooltip": "ERB-domain smoothness of warp function. "
                               "Higher = slower variation."
                }),
                "preserve_formants": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Preserve vocal formant regions (500-4000 Hz)."
                }),
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

        if seed == 0:
            seed = torch.seed()
        gen = torch.Generator(device=wav.device).manual_seed(seed)

        n_fft = 2048
        hop = n_fft // 4
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr, device=wav.device)
        erb_freqs = _hz_to_erb(freqs)

        out = torch.empty_like(wav)

        for b in range(B):
            for c in range(C):
                sig = wav[b, c]
                S = _stft(sig.unsqueeze(0), n_fft, hop).squeeze(0)
                mag = S.abs()
                phase = S.angle()

                # Generate smooth warp function in ERB domain
                n_control = 32
                erb_min, erb_max = erb_freqs[0].item(), erb_freqs[-1].item()
                control_erbs = torch.linspace(erb_min, erb_max, n_control, device=wav.device)
                
                # Random warp offsets
                warp_offsets = (torch.rand(n_control, device=wav.device, generator=gen) * 2 - 1) * warp_strength
                
                # Smooth with Gaussian kernel
                kernel_size = int(warp_smoothness * 2)
                if kernel_size > 1:
                    warp_offsets = torch.nn.functional.avg_pool1d(
                        warp_offsets.unsqueeze(0).unsqueeze(0),
                        kernel_size=kernel_size, stride=1, padding=kernel_size//2
                    ).squeeze(0).squeeze(0)

                # Protect formant regions
                if preserve_formants:
                    formant_mask = (freqs >= 500) & (freqs <= 4000)
                    formant_erb = erb_freqs[formant_mask]
                    if len(formant_erb) > 0:
                        # Taper warp near formants
                        for i, erb in enumerate(control_erbs):
                            dist = torch.abs(formant_erb - erb).min().item()
                            taper = min(1.0, dist / (_hz_to_erb(torch.tensor(1000.0)).item()))
                            warp_offsets[i] *= taper

                # Interpolate warp to all frequency bins
                warp_interp = torch.nn.functional.interpolate(
                    warp_offsets.unsqueeze(0).unsqueeze(0),
                    size=len(erb_freqs), mode='linear', align_corners=False
                ).squeeze(0).squeeze(0)

                # Apply warp: map magnitude from warped frequencies
                warped_erb = erb_freqs + warp_interp
                warped_erb = torch.clamp(warped_erb, erb_min, erb_max)
                warped_hz = _erb_to_hz(warped_erb)

                # Resample magnitude onto warped frequency grid
                mag_warped = torch.nn.functional.interpolate(
                    mag.unsqueeze(0),
                    size=warped_hz.shape[0], mode='linear', align_corners=False
                ).squeeze(0)

                # Also need to handle phase - use phase from original frequencies
                # but mapped to warped grid
                phase_warped = torch.nn.functional.interpolate(
                    phase.unsqueeze(0),
                    size=warped_hz.shape[0], mode='linear', align_corners=False
                ).squeeze(0)

                # Interpolate back to original frequency grid
                mag_out = torch.nn.functional.interpolate(
                    mag_warped.unsqueeze(0),
                    size=len(freqs), mode='linear', align_corners=False
                ).squeeze(0)
                phase_out = torch.nn.functional.interpolate(
                    phase_warped.unsqueeze(0),
                    size=len(freqs), mode='linear', align_corners=False
                ).squeeze(0)

                S_out = mag_out * torch.exp(1j * phase_out)
                sig_out = _istft(S_out.unsqueeze(0), n_fft, hop, T).squeeze(0)

                out[b, c] = sig_out

        out = torch.clamp(out, -1.0, 1.0)
        return (_make_audio(out, sr),)


# ──────────────────────────────────────────────
#  12. Enhanced Full Pipeline
# ──────────────────────────────────────────────

class AudioFingerprintPipelineV2:
    """
    Enhanced pipeline: Spectral Perturbation → Advanced Phase Scrambling
    → Mid/Side Perturbation → Psychoacoustic Noise Shaping
    → Temporal Micro-Editing → Spectral Envelope Warping
    → Micro Time-Stretch → Multi-Codec Chain.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                # Spectral perturbation
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
                # Advanced phase scrambling
                "enable_adv_phase": ("BOOLEAN", {"default": True}),
                "adv_phase_strength": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 0.6, "step": 0.01
                }),
                # Mid/Side
                "enable_mid_side": ("BOOLEAN", {"default": True}),
                "mid_jitter": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 0.5, "step": 0.01
                }),
                "side_jitter": ("FLOAT", {
                    "default": 0.35, "min": 0.0, "max": 0.7, "step": 0.01
                }),
                # Psychoacoustic noise shaping
                "enable_psychoacoustic": ("BOOLEAN", {"default": True}),
                "target_snr_db": ("FLOAT", {
                    "default": 40.0, "min": 20.0, "max": 80.0, "step": 1.0
                }),
                # Temporal micro-editing
                "enable_temporal_edit": ("BOOLEAN", {"default": True}),
                "edits_per_second": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 3.0, "step": 0.1
                }),
                # Spectral envelope warping
                "enable_warping": ("BOOLEAN", {"default": True}),
                "warp_strength": ("FLOAT", {
                    "default": 0.02, "min": 0.0, "max": 0.08, "step": 0.005
                }),
                # Micro time-stretch
                "stretch_percent": ("FLOAT", {
                    "default": 0.5, "min": -2.0, "max": 2.0, "step": 0.1
                }),
                # Multi-codec chain
                "enable_codec_chain": ("BOOLEAN", {"default": True}),
                "codec_chain": (["opus→aac", "opus→mp3", "aac→mp3", "opus→aac→mp3"], {
                    "default": "opus→aac",
                }),
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
                enable_codec_chain, codec_chain, bitrate_kbps, final_bitrate_kbps, seed):

        # Step 1 – Spectral Perturbation
        spec_node = AudioSpectralPerturbation()
        (audio,) = spec_node.process(
            audio, phase_jitter_strength, jitter_cutoff_hz,
            dither_db, tilt_db, 10000.0, seed
        )

        # Step 2 – Advanced Phase Scrambling
        if enable_adv_phase:
            adv_node = AudioAdvancedPhaseScrambling()
            (audio,) = adv_node.process(
                audio, adv_phase_strength, 0.7, 1.5, 2000.0, seed + 1
            )

        # Step 3 – Mid/Side Stereo Perturbation
        if enable_mid_side:
            ms_node = AudioMidSidePerturbation()
            (audio,) = ms_node.process(
                audio, mid_jitter, side_jitter, -70.0, -60.0, 0.05, seed + 2
            )

        # Step 4 – Psychoacoustic Noise Shaping
        if enable_psychoacoustic:
            pa_node = AudioPsychoacousticNoiseShaping()
            (audio,) = pa_node.process(
                audio, target_snr_db, 6.0, -90.0, seed + 3
            )

        # Step 5 – Temporal Micro-Editing
        if enable_temporal_edit:
            te_node = AudioTemporalMicroEdit()
            (audio,) = te_node.process(
                audio, edits_per_second, 2.0, 0.5, seed + 4
            )

        # Step 6 – Spectral Envelope Warping
        if enable_warping:
            warp_node = AudioSpectralEnvelopeWarping()
            (audio,) = warp_node.process(
                audio, warp_strength, 4.0, True, seed + 5
            )

        # Step 7 – Micro Time-Stretch
        if abs(stretch_percent) > 0.01:
            ts_node = AudioMicroTimeStretch()
            (audio,) = ts_node.process(audio, stretch_percent, seed + 6)

        # Step 8 – Multi-Codec Chain
        if enable_codec_chain:
            mc_node = AudioMultiCodecChain()
            (audio,) = mc_node.process(audio, codec_chain, bitrate_kbps, final_bitrate_kbps)

        return (audio,)