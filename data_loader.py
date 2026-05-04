"""
data_loader.py — EEG Data Ingestion & Synthetic Generator
----------------------------------------------------------
Supports:
  • Loading EEG from CSV files
  • Generating realistic synthetic multi-channel EEG
    (alpha / beta / theta / delta / gamma rhythms)
  • Injecting seizure-like and burst events for demo purposes
"""

import numpy as np
import pandas as pd
from pathlib import Path
from utils import EEG_CHANNELS, ts_now


# ─── Sampling constants ─────────────────────────────────────────────────────────
FS = 256           # Sampling frequency (Hz)
DURATION_S = 60    # Default signal duration in seconds


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Synthetic EEG Generator
# ═══════════════════════════════════════════════════════════════════════════════

def _band_signal(t: np.ndarray, freqs: list[float], amp: float) -> np.ndarray:
    """Superpose sinusoids within a frequency band with random phases."""
    signal = np.zeros(len(t))
    for f in freqs:
        phase = np.random.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * f * t + phase)
    return signal


def _inject_seizure(signal: np.ndarray, fs: int, onset: float,
                    duration: float = 4.0) -> np.ndarray:
    """Inject a seizure-like burst (high-amplitude, high-frequency) into a 1-D signal."""
    n     = len(signal)
    start = int(onset * fs)
    end   = int((onset + duration) * fs)
    # Clamp to valid range
    start = max(0, min(start, n - 1))
    end   = max(start + 1, min(end, n))
    seg_len = end - start
    t_seg = np.linspace(0, duration, seg_len)
    burst = 150 * np.sin(2 * np.pi * 8 * t_seg) * np.exp(-0.5 * (t_seg - duration / 2) ** 2)
    signal = signal.copy()
    signal[start:end] += burst
    return signal


def _inject_sharp_spike(signal: np.ndarray, fs: int, onset: float) -> np.ndarray:
    """Inject an isolated sharp spike (interictal discharge)."""
    n     = len(signal)
    idx   = min(int(onset * fs), n - 1)
    width = 10   # samples
    s     = max(0, idx - width)
    e     = min(n, idx + width)
    actual_width = e - s
    if actual_width < 1:
        return signal
    spike = np.zeros(n)
    spike[s:e] = np.linspace(0, 200, actual_width)
    signal = signal.copy()
    signal += spike * np.exp(-np.linspace(0, 5, n))
    return signal


def generate_synthetic_eeg(
    duration_s: int = DURATION_S,
    fs: int = FS,
    channels: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic multi-channel synthetic EEG DataFrame.

    Returns a DataFrame with columns:
        timestamp, channel, value, label
    """
    if channels is None:
        channels = EEG_CHANNELS[:8]   # default: first 8 channels

    rng  = np.random.default_rng(seed)
    t    = np.arange(0, duration_s, 1.0 / fs)
    n    = len(t)
    rows = []

    # Planned event windows (to be labelled)
    seizure_channels = rng.choice(channels, size=2, replace=False)
    seizure_onsets   = [12.0, 38.0]                # seconds
    spike_channels   = rng.choice(channels, size=3, replace=False)
    spike_onsets     = [5.0, 22.0, 50.0]

    for ch in channels:
        # --- Compose EEG bands ---
        alpha = _band_signal(t, [8, 9, 10, 11, 12], amp=30)
        beta  = _band_signal(t, [14, 18, 22, 26, 30], amp=15)
        theta = _band_signal(t, [4, 5, 6, 7], amp=20)
        delta = _band_signal(t, [0.5, 1, 2, 3], amp=25)
        gamma = _band_signal(t, [32, 40, 50, 60], amp=8)
        noise = rng.normal(0, 5, n)

        sig = alpha + beta + theta + delta + gamma + noise

        # --- Inject events per channel ---
        labels = np.full(n, "background", dtype=object)

        for onset in seizure_onsets:
            if ch in seizure_channels:
                sig = _inject_seizure(sig, fs, onset, duration=4.0)
                s, e = int(onset * fs), int((onset + 4) * fs)
                labels[s:e] = "seizure"

        for onset in spike_onsets:
            if ch in spike_channels:
                sig = _inject_sharp_spike(sig, fs, onset)
                s, e = max(0, int(onset * fs) - 20), int(onset * fs) + 20
                labels[s:e] = "spike"

        # Build per-sample timestamps
        timestamps = pd.date_range("2024-01-01", periods=n, freq=f"{int(1e9/fs)}ns")

        ch_df = pd.DataFrame({
            "timestamp": timestamps,
            "channel":   ch,
            "value":     sig,
            "label":     labels,
        })
        rows.append(ch_df)

    df = pd.concat(rows, ignore_index=True).sort_values("timestamp")
    df["timestamp"] = df["timestamp"].astype(str)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CSV Loaders  (auto-detecting: wide OR long format)
# ═══════════════════════════════════════════════════════════════════════════════

LONG_REQUIRED = {"timestamp", "channel", "value"}


# ── Format sniffers ────────────────────────────────────────────────────────────

def _is_long_format(df: pd.DataFrame) -> bool:
    """
    Return True if the DataFrame looks like long-format EEG
    (has 'timestamp', 'channel', 'value' columns).
    """
    cols = set(df.columns.str.lower())
    return LONG_REQUIRED.issubset(cols)


def _infer_fs(time_col: pd.Series) -> int:
    """
    Infer sampling frequency from a time column.
    Handles numeric seconds, milliseconds, and datetime strings.
    Returns an integer Hz estimate (default 256 if inference fails).
    """
    try:
        times = pd.to_numeric(time_col, errors="coerce").dropna()
        if len(times) >= 2:
            dt = float(times.iloc[1] - times.iloc[0])
            if dt <= 0:
                return FS
            # Detect unit: if values look like ms (dt >> 1) treat as ms
            if dt > 0.5:        # probably milliseconds
                return max(1, round(1000 / dt))
            else:               # probably seconds
                return max(1, round(1.0 / dt))
    except Exception:
        pass
    return FS   # fallback


# ── Wide-format loader ─────────────────────────────────────────────────────────

def load_eeg_csv_wide(
    filepath: str | Path,
    time_col:  str | None = None,
    fs_hint:   int | None = None,
) -> tuple[pd.DataFrame, int]:
    """
    Load a **wide-format** EEG CSV where:
      • Each row   = one time sample
      • Each column = one EEG channel (or the time axis)

    The function auto-detects which column (if any) holds time values.
    If no dedicated time column exists, a synthetic timestamp is created
    at the inferred (or hinted) sampling rate.

    Returns
    -------
    df  : long-format DataFrame with columns (timestamp, channel, value, label)
    fs  : inferred or hinted sampling frequency (Hz)
    """
    raw = pd.read_csv(filepath, header=0)
    raw.columns = [str(c).strip() for c in raw.columns]

    # --- Identify the time column ---
    time_candidates = [c for c in raw.columns
                       if c.lower() in ("time", "t", "timestamp", "times", "sample", "index")]

    t_col = None
    if time_col and time_col in raw.columns:
        t_col = time_col
    elif time_candidates:
        t_col = time_candidates[0]
    else:
        # Check if the first column is numeric-ish and monotonically increasing
        first = raw.columns[0]
        numeric_vals = pd.to_numeric(raw[first], errors="coerce")
        if numeric_vals.notna().mean() > 0.9 and numeric_vals.is_monotonic_increasing:
            t_col = first

    # --- Determine channel columns ---
    if t_col:
        ch_cols = [c for c in raw.columns if c != t_col]
        time_values = pd.to_numeric(raw[t_col], errors="coerce")
        fs = fs_hint or _infer_fs(time_values)
    else:
        ch_cols = list(raw.columns)
        fs = fs_hint or FS
        time_values = None

    # --- Coerce channel data to numeric (drop non-numeric columns) ---
    valid_ch_cols = []
    for c in ch_cols:
        numeric = pd.to_numeric(raw[c], errors="coerce")
        if numeric.notna().mean() > 0.5:   # at least 50% numeric
            raw[c] = numeric.fillna(0.0)
            valid_ch_cols.append(c)

    if not valid_ch_cols:
        raise ValueError(
            "No numeric channel columns found in wide-format CSV. "
            "Check that amplitude columns contain numbers."
        )

    n_samples = len(raw)

    # --- Build synthetic timestamps (seconds) ---
    if time_values is not None and time_values.notna().all():
        # Normalise to seconds if needed
        tv = time_values.to_numpy(dtype=float)
        dt = tv[1] - tv[0] if len(tv) > 1 else 1.0 / fs
        if dt > 0.5:    # milliseconds → convert
            tv = tv / 1000.0
        timestamps = [f"{v:.6f}" for v in tv]
    else:
        tv = np.arange(n_samples) / fs
        timestamps = [f"{v:.6f}" for v in tv]

    # --- Assign channel names: use column header if it looks like an EEG label,
    #     otherwise fall back to CH_00, CH_01, … ---
    eeg_labels = {c.upper() for c in EEG_CHANNELS}
    assigned_names = []
    for i, col in enumerate(valid_ch_cols):
        if col.upper() in eeg_labels:
            assigned_names.append(col.upper())
        else:
            # Try to use the column name directly as channel label
            clean = col.strip().replace(" ", "_")
            assigned_names.append(clean if clean else f"CH_{i:02d}")

    # --- Melt to long format ---
    rows = []
    for ch_name, col in zip(assigned_names, valid_ch_cols):
        rows.append(pd.DataFrame({
            "timestamp": timestamps,
            "channel":   ch_name,
            "value":     raw[col].to_numpy(dtype=float),
            "label":     "unknown",
        }))

    long_df = pd.concat(rows, ignore_index=True)
    return long_df, fs


# ── Long-format loader ─────────────────────────────────────────────────────────

def load_eeg_csv_long(filepath: str | Path) -> tuple[pd.DataFrame, int]:
    """
    Load a **long-format** EEG CSV with columns: timestamp, channel, value, [label].
    Returns (df, fs) where fs is inferred from the timestamp column.
    """
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = LONG_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Long-format CSV missing columns: {missing}")

    if "label" not in df.columns:
        df["label"] = "unknown"

    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)

    # Infer fs from the first channel's timestamps
    ch0 = df["channel"].iloc[0]
    t0  = df[df["channel"] == ch0]["timestamp"].reset_index(drop=True)
    fs  = _infer_fs(t0)

    return df, fs


# ── Auto-detecting entry point ─────────────────────────────────────────────────

def load_eeg_csv(
    filepath: str | Path,
    time_col:  str | None = None,
    fs_hint:   int | None = None,
) -> tuple[pd.DataFrame, int]:
    """
    **Universal EEG CSV loader** — auto-detects wide vs long format.

    Wide format  (your dataset):  rows = samples, columns = channels
    Long format  (internal):      columns = timestamp, channel, value

    Returns
    -------
    df  : long-format DataFrame (timestamp, channel, value, label)
    fs  : sampling frequency in Hz
    """
    raw = pd.read_csv(filepath, nrows=3)   # peek at headers only
    raw.columns = [c.strip().lower() for c in raw.columns]

    if _is_long_format(raw):
        return load_eeg_csv_long(filepath)
    else:
        return load_eeg_csv_wide(filepath, time_col=time_col, fs_hint=fs_hint)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Sample CSV Writer (for first-run bootstrapping)
# ═══════════════════════════════════════════════════════════════════════════════

def write_sample_csv(path: str | Path = "sample_data/eeg_sample.csv"):
    """Write a small sample EEG CSV that ships with the project."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_eeg(duration_s=30, channels=EEG_CHANNELS[:4], seed=0)
    df.to_csv(path, index=False)
    print(f"[data_loader] Sample EEG written -> {path}")


def write_sample_csv_wide(path: str | Path = "sample_data/eeg_sample_wide.csv"):
    """Write a wide-format sample CSV for testing the wide loader."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = EEG_CHANNELS[:4]
    df_long, _ = load_eeg_csv("sample_data/eeg_sample.csv")
    pivot = df_long.pivot_table(index="timestamp", columns="channel", values="value", aggfunc="first")
    pivot.reset_index(inplace=True)
    pivot.rename(columns={"timestamp": "time"}, inplace=True)
    pivot.to_csv(path, index=False)
    print(f"[data_loader] Wide-format sample written -> {path}")


# ─── Module self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    write_sample_csv()
    df, fs = load_eeg_csv("sample_data/eeg_sample.csv")
    print(f"[long] Shape={df.shape} | fs={fs} Hz | Channels={list(df.channel.unique())}")
    print(df.head(3).to_string())
