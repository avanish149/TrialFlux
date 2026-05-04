"""
event_detector.py — EEG Event Detection Pipeline
-------------------------------------------------
Processes a single channel's time-series in sliding windows and labels
each window as HIGH or LOW priority using a battery of clinical heuristics:

  1. Z-score deviation from rolling baseline
  2. Peak-to-peak amplitude
  3. Variance threshold
  4. Spike density (count of samples > N×σ)
  5. Band-power ratio (gamma / delta) as a seizure proxy
  6. Morphological sharpness (second-derivative gradient)

A priority score [0-100] is produced for every window.
Windows ≥ priority_threshold (default 60) are HIGH priority.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy import signal as sp_signal
from utils import zscore

# --- Compatibility for NumPy 2.0+ ---
if hasattr(np, "trapezoid"):
    _trapz = np.trapezoid
else:
    _trapz = np.trapz


# ─── Window configuration ───────────────────────────────────────────────────────
WINDOW_S     = 2      # seconds per window
STEP_S       = 1      # step / overlap in seconds
FS           = 256    # expected sampling frequency


@dataclass
class DetectedWindow:
    """One analysis window for a single EEG channel."""
    window_id:      int
    channel:        str
    t_start:        float          # seconds from recording start
    t_end:          float
    priority:       str            # "HIGH" | "LOW"
    priority_score: float          # 0-100
    event_types:    list[str]      # e.g. ["spike", "burst"]
    stats:          dict           # detailed feature dict
    values:         np.ndarray = field(repr=False)   # raw signal in window


# ═══════════════════════════════════════════════════════════════════════════════
# Feature extractors
# ═══════════════════════════════════════════════════════════════════════════════

def _zscore_max(window: np.ndarray) -> float:
    """Maximum absolute z-score within the window."""
    z = zscore(window)
    return float(np.max(np.abs(z)))


def _peak_to_peak(window: np.ndarray) -> float:
    return float(window.max() - window.min())


def _variance(window: np.ndarray) -> float:
    return float(np.var(window))


def _spike_density(window: np.ndarray, sigma_thresh: float = 3.0) -> float:
    """Fraction of samples that are >sigma_thresh standard deviations from mean."""
    sigma = window.std() + 1e-9
    return float(np.mean(np.abs(window - window.mean()) > sigma_thresh * sigma))


def _band_power(window: np.ndarray, fs: int, low: float, high: float) -> float:
    """Estimate power in a frequency band using Welch's method."""
    n = len(window)
    if n < 16:
        return 0.0
    nperseg = min(n, 64)
    freqs, psd = sp_signal.welch(window, fs=fs, nperseg=nperseg)
    band_mask = (freqs >= low) & (freqs <= high)
    return float(_trapz(psd[band_mask], freqs[band_mask]))


def _sharpness(window: np.ndarray) -> float:
    """Mean absolute second derivative — captures sharp morphological changes."""
    if len(window) < 3:
        return 0.0
    return float(np.mean(np.abs(np.diff(window, n=2))))


# ═══════════════════════════════════════════════════════════════════════════════
# Priority scorer
# ═══════════════════════════════════════════════════════════════════════════════

def _score_window(window: np.ndarray, fs: int) -> tuple[float, list[str], dict]:
    """
    Compute a composite priority score [0-100] and label detected event types.
    Returns (score, event_types, stats_dict).
    """
    z_max     = _zscore_max(window)
    ptp       = _peak_to_peak(window)
    var       = _variance(window)
    spike_den = _spike_density(window)
    gamma_pow = _band_power(window, fs, 30, 80)
    delta_pow = _band_power(window, fs, 0.5, 4) + 1e-9
    sharp     = _sharpness(window)
    gd_ratio  = gamma_pow / delta_pow

    score = 0.0
    events: list[str] = []

    # --- Rule contributions (each rule adds up to a defined weight) ---
    # Z-score deviation (weight 30)
    if z_max > 5:
        score += 30; events.append("extreme_deviation")
    elif z_max > 3:
        score += 18; events.append("deviation")
    elif z_max > 2:
        score += 8

    # Spike density (weight 25)
    if spike_den > 0.15:
        score += 25; events.append("spike_burst")
    elif spike_den > 0.08:
        score += 15; events.append("spike")
    elif spike_den > 0.03:
        score += 6

    # Band-power ratio — seizure proxy (weight 20)
    if gd_ratio > 3.0:
        score += 20; events.append("seizure_like")
    elif gd_ratio > 1.5:
        score += 10; events.append("high_gamma")

    # Morphological sharpness (weight 15)
    if sharp > 50:
        score += 15; events.append("sharp_transient")
    elif sharp > 20:
        score += 8

    # Variance (weight 10)
    if var > 5000:
        score += 10; events.append("high_variance")
    elif var > 1000:
        score += 4

    score = min(score, 100.0)

    stats = {
        "z_max":      round(z_max, 3),
        "peak2peak":  round(ptp, 3),
        "variance":   round(var, 3),
        "spike_den":  round(spike_den, 4),
        "gd_ratio":   round(gd_ratio, 4),
        "sharpness":  round(sharp, 4),
    }
    return score, events, stats


# ═══════════════════════════════════════════════════════════════════════════════
# Main detector
# ═══════════════════════════════════════════════════════════════════════════════

def detect_events(
    channel_data: np.ndarray,
    channel_name: str,
    fs: int = FS,
    window_s: float = WINDOW_S,
    step_s: float = STEP_S,
    priority_threshold: float = 60.0,
) -> list[DetectedWindow]:
    """
    Slide a window over `channel_data` and return a list of DetectedWindow objects.

    Parameters
    ----------
    channel_data      : 1-D array of EEG voltage values
    channel_name      : channel label string
    fs                : sampling frequency (Hz)
    window_s          : window duration (seconds)
    step_s            : step size (seconds)
    priority_threshold: score ≥ threshold → HIGH priority
    """
    # Automatically shrink window if data is too short
    duration_s = len(channel_data) / fs
    if duration_s < window_s:
        window_s = duration_s
        step_s = duration_s

    win_samples  = int(window_s * fs)
    step_samples = max(1, int(step_s * fs))
    n            = len(channel_data)
    windows: list[DetectedWindow] = []
    win_id = 0

    for start in range(0, n - win_samples + 1, step_samples):
        end    = start + win_samples
        window = channel_data[start:end]

        score, events, stats = _score_window(window, fs)

        windows.append(DetectedWindow(
            window_id      = win_id,
            channel        = channel_name,
            t_start        = start / fs,
            t_end          = end / fs,
            priority       = "HIGH" if score >= priority_threshold else "LOW",
            priority_score = round(score, 2),
            event_types    = events,
            stats          = stats,
            values         = window,
        ))
        win_id += 1

    return windows


def detect_all_channels(
    df: pd.DataFrame,
    fs: int = FS,
    window_s: float = WINDOW_S,
    step_s: float = STEP_S,
    priority_threshold: float = 60.0,
) -> list[DetectedWindow]:
    """Run event detection on every channel in a TrialFlux EEG DataFrame."""
    all_windows: list[DetectedWindow] = []
    for ch in df["channel"].unique():
        vals = df[df["channel"] == ch]["value"].to_numpy()
        wins = detect_events(vals, ch, fs, window_s, step_s, priority_threshold)
        all_windows.extend(wins)
    return all_windows


def summarise_events(windows: list[DetectedWindow]) -> pd.DataFrame:
    """Convert detected windows to a tidy summary DataFrame."""
    rows = []
    for w in windows:
        rows.append({
            "window_id":     w.window_id,
            "channel":       w.channel,
            "t_start":       w.t_start,
            "t_end":         w.t_end,
            "priority":      w.priority,
            "score":         w.priority_score,
            "events":        ", ".join(w.event_types) if w.event_types else "—",
            **w.stats,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=[
            "window_id", "channel", "t_start", "t_end", "priority", 
            "score", "events", "p2p_amp", "rms", "gamma_delta_ratio"
        ])
    return df


# ─── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import generate_synthetic_eeg
    df   = generate_synthetic_eeg(duration_s=30, channels=["Fp1", "C3"])
    wins = detect_all_channels(df)
    summary = summarise_events(wins)
    hi = summary[summary.priority == "HIGH"]
    print(f"Total windows: {len(summary)}  |  HIGH priority: {len(hi)}")
    print(hi[["channel","t_start","t_end","score","events"]].to_string(index=False))
