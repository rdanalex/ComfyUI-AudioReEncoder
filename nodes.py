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

logger = logging.getLogger("AudioFingerprintRemover")

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
#  4. Full Pipeline  (all-in-one convenience)
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