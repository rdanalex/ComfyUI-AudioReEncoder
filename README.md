# ComfyUI-AudioReEncoder

Removes codec-level and spectral fingerprints from AI-generated audio
so downstream AI-music detectors do not attribute the output to a
specific upstream generator (e.g. Suno, Udio, etc.).

Designed and tested with **YuE / YuE2** but works with any audio
generation pipeline that outputs the ComfyUI `AUDIO` type
(`[batch, channels, samples]`).

## What it does

| Node | Purpose |
|---|---|
| **Audio Spectral Perturbation** | High-frequency phase jitter, broadband dither, gentle spectral tilt. Breaks phase-coherence and noise-floor signatures. |
| **Audio Micro Time-Stretch** | ±0.1–2 % pitch-preserving time-stretch via phase vocoder. Shifts temporal statistics. |
| **Audio Codec Re-encode** | FFmpeg round-trip through Opus / AAC / MP3. Destroys neural-codec reconstruction artifacts. |
| **Audio Fingerprint Remover (Full Pipeline)** | Chains all three in one node. |

All processing is **audibly transparent** at default settings.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/rdanalex/ComfyUI-AudioReEncoder.git
cd ComfyUI-AudioReEncoder
pip install -r requirements.txt
```

> **Note:** The *Codec Re-encode* node requires **FFmpeg** on your
> system PATH. Install via `apt install ffmpeg`, `brew install ffmpeg`,
> or download from [ffmpeg.org](https://ffmpeg.org).

Restart ComfyUI. Nodes appear under **audio → fingerprint_removal**.

## Quick-start workflow

```
YuE2 Audio Out
  → Audio Fingerprint Remover (Full Pipeline)   ← single node
  → SaveAudio / PreviewAudio
```

Or wire the three nodes individually for fine-grained control.

## Recommended (inaudible) settings

| Parameter | Value | Notes |
|---|---|---|
| `phase_jitter_strength` | 0.25 | >0.5 → audible phasing |
| `jitter_cutoff_hz` | 8000 | <5000 → audible smearing |
| `dither_db` | -65 | well below room noise floor |
| `tilt_db` | 0.0 – 1.0 | ±2 dB max before audible |
| `stretch_percent` | 0.5 | >2 % → transient smearing |
| `codec` | opus | 320 kbps is transparent |

## Quality notes

The defaults are intended to be transparent for typical program material,
but the result depends on the source and settings. Listen-check output when
using stronger jitter, tilt, dither, or time-stretch values.

## How it works (technical)

AI-generated audio carries a *codec fingerprint*: consistent
reconstruction artifacts from the neural audio codec used during
training/inference. Detectors learn these artifacts as statistical
features in the spectro-temporal domain.

This pack breaks three layers of that fingerprint:

1. **Phase coherence** – randomised above 8 kHz where human phase
   sensitivity is negligible.
2. **Noise-floor signature** – a -65 dBFS dither breaks the
   "perfectly clean reconstruction" pattern.
3. **Codec quantisation grid** – re-encoding through a completely
   different codec (Opus MDCT) destroys the original codec's
   reconstruction texture.

## License

MIT