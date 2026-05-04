"""
compressor.py — Adaptive EEG Compression
-----------------------------------------
Strategy:
  • HIGH-priority windows  → lossless (store full signal, ratio ≈ 1.0)
  • LOW-priority windows   → lossy via:
        - DCT truncation (discard high-frequency coefficients)
        - Downsampling + linear interpolation on decode

Each segment is packaged as a CompressedSegment with a manifest entry.
"""

import numpy as np
import json
import hashlib
from dataclasses import dataclass, field, asdict
from scipy.fft import dct, idct
from scipy.interpolate import interp1d
from event_detector import DetectedWindow


# ─── Compression levels ─────────────────────────────────────────────────────────
LOSSLESS_KEEP_RATIO = 1.00    # keep 100 % of DCT coefficients
LOW_KEEP_RATIO      = 0.30    # keep 30 % of DCT coefficients (lossy)
DOWNSAMPLE_FACTOR   = 4       # downsample factor for "very low priority" option


@dataclass
class CompressedSegment:
    """One compressed window / segment ready for transmission."""
    segment_id:      str       # unique ID
    window_id:       int
    channel:         str
    t_start:         float
    t_end:           float
    priority:        str       # HIGH | LOW
    priority_score:  float
    event_types:     list[str]

    # Compression metadata
    method:          str       # "lossless_dct" | "lossy_dct"
    keep_ratio:      float
    n_original:      int       # original sample count
    n_stored:        int       # stored coefficient count
    compression_ratio: float

    # Payload
    coefficients:    np.ndarray = field(repr=False)   # DCT coefficients stored

    # Integrity (filled in by integrity.py)
    segment_hash:    str = ""
    chain_hash:      str = ""


def _compress_window(window: np.ndarray, keep_ratio: float) -> np.ndarray:
    """Apply DCT and keep only the first `keep_ratio` fraction of coefficients."""
    coeffs    = dct(window, norm="ortho")
    n_keep    = max(1, int(len(coeffs) * keep_ratio))
    truncated = coeffs.copy()
    truncated[n_keep:] = 0.0      # zero out high-freq components
    return truncated[:n_keep]


def _decompress_window(stored_coeffs: np.ndarray, n_original: int) -> np.ndarray:
    """Reconstruct original-length signal from stored DCT coefficients."""
    padded = np.zeros(n_original)
    n_keep = min(len(stored_coeffs), n_original)
    padded[:n_keep] = stored_coeffs[:n_keep]
    return idct(padded, norm="ortho")


# ═══════════════════════════════════════════════════════════════════════════════
# Main compress / decompress API
# ═══════════════════════════════════════════════════════════════════════════════

def compress_windows(
    windows: list[DetectedWindow],
    high_keep: float = LOSSLESS_KEEP_RATIO,
    low_keep:  float = LOW_KEEP_RATIO,
) -> list[CompressedSegment]:
    """
    Compress a list of DetectedWindow objects into CompressedSegment packets.

    Parameters
    ----------
    windows   : list of DetectedWindow from event_detector
    high_keep : fraction of DCT coefficients to keep for HIGH-priority windows
    low_keep  : fraction of DCT coefficients to keep for LOW-priority windows
    """
    segments: list[CompressedSegment] = []

    for win in windows:
        keep_ratio = high_keep if win.priority == "HIGH" else low_keep
        method     = "lossless_dct" if win.priority == "HIGH" else "lossy_dct"

        coeffs           = _compress_window(win.values, keep_ratio)
        n_original       = len(win.values)
        n_stored         = len(coeffs)
        compression_ratio = n_original / n_stored

        seg_id = f"{win.channel}_{win.window_id:05d}"

        seg = CompressedSegment(
            segment_id        = seg_id,
            window_id         = win.window_id,
            channel           = win.channel,
            t_start           = win.t_start,
            t_end             = win.t_end,
            priority          = win.priority,
            priority_score    = win.priority_score,
            event_types       = win.event_types,
            method            = method,
            keep_ratio        = keep_ratio,
            n_original        = n_original,
            n_stored          = n_stored,
            compression_ratio = round(compression_ratio, 3),
            coefficients      = coeffs,
        )
        segments.append(seg)

    return segments


def decompress_segment(seg: CompressedSegment) -> np.ndarray:
    """Reconstruct the time-domain signal from a CompressedSegment."""
    return _decompress_window(seg.coefficients, seg.n_original)


def decompress_all(segments: list[CompressedSegment]) -> dict[str, np.ndarray]:
    """
    Reconstruct per-channel signals by decompressing and concatenating segments.
    Returns dict: channel -> reconstructed time series
    """
    by_channel: dict[str, list[tuple[float, np.ndarray]]] = {}

    for seg in segments:
        ch = seg.channel
        recon = decompress_segment(seg)
        by_channel.setdefault(ch, []).append((seg.t_start, recon))

    result: dict[str, np.ndarray] = {}
    for ch, chunks in by_channel.items():
        chunks.sort(key=lambda x: x[0])
        result[ch] = np.concatenate([c[1] for c in chunks])

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Manifest
# ═══════════════════════════════════════════════════════════════════════════════

def build_manifest(segments: list[CompressedSegment]) -> list[dict]:
    """
    Build a JSON-serialisable manifest for all segments.
    The manifest is what a remote server would check for provenance.
    """
    manifest = []
    for seg in segments:
        manifest.append({
            "segment_id":        seg.segment_id,
            "window_id":         seg.window_id,
            "channel":           seg.channel,
            "t_start":           seg.t_start,
            "t_end":             seg.t_end,
            "priority":          seg.priority,
            "priority_score":    seg.priority_score,
            "event_flag":        len(seg.event_types) > 0,
            "event_types":       seg.event_types,
            "method":            seg.method,
            "keep_ratio":        seg.keep_ratio,
            "n_original":        seg.n_original,
            "n_stored":          seg.n_stored,
            "compression_ratio": seg.compression_ratio,
            "segment_hash":      seg.segment_hash,
            "chain_hash":        seg.chain_hash,
        })
    return manifest


# ─── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import generate_synthetic_eeg
    from event_detector import detect_all_channels

    df      = generate_synthetic_eeg(duration_s=20, channels=["Fp1"])
    windows = detect_all_channels(df)
    segs    = compress_windows(windows)

    total_orig   = sum(s.n_original for s in segs)
    total_stored = sum(s.n_stored   for s in segs)
    print(f"Segments: {len(segs)}")
    print(f"Overall compression ratio: {total_orig / (total_stored + 1):.2f}×")
    hi = [s for s in segs if s.priority == "HIGH"]
    lo = [s for s in segs if s.priority == "LOW"]
    print(f"HIGH: {len(hi)}  LOW: {len(lo)}")
