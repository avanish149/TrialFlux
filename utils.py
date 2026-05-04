"""
utils.py — Shared utilities for TrialFlux
Contains color maps, formatting helpers, and common constants.
"""

import numpy as np
import pandas as pd
from datetime import datetime

# ─── Clinical colour palette ───────────────────────────────────────────────────
COLORS = {
    "primary": "#6C63FF",
    "accent": "#00D4AA",
    "danger": "#FF4E5B",
    "warning": "#FFC947",
    "success": "#36D399",
    "muted": "#8892A4",
    "bg_dark": "#0F1117",
    "card": "#1A1D27",
}

# ─── EEG channel names (10-20 system) ──────────────────────────────────────────
EEG_CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
                 "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz"]

# ─── Classification labels ──────────────────────────────────────────────────────
CLASS_BIOLOGICAL    = "Biological Event"
CLASS_CORRUPTION    = "Network Corruption"
CLASS_TAMPER        = "Tampering / Fabrication"
CLASS_CLEAN         = "Clean"

# ─── Verdict badge colours ──────────────────────────────────────────────────────
VERDICT_COLOR = {
    CLASS_CLEAN:      COLORS["success"],
    CLASS_BIOLOGICAL: COLORS["accent"],
    CLASS_CORRUPTION: COLORS["warning"],
    CLASS_TAMPER:     COLORS["danger"],
}


def ts_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def snr_db(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Signal-to-Noise Ratio in dB between two arrays."""
    signal_power = np.mean(original ** 2)
    noise_power  = np.mean((original - reconstructed) ** 2) + 1e-12
    return float(10 * np.log10(signal_power / noise_power))


def rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr ** 2)))


def zscore(arr: np.ndarray) -> np.ndarray:
    mu, sigma = arr.mean(), arr.std() + 1e-9
    return (arr - mu) / sigma


def percent_change(original: float, new: float) -> float:
    if original == 0:
        return 0.0
    return (new - original) / abs(original) * 100


def integrity_badge(score: float) -> str:
    """Return a textual integrity badge given a 0-100 score."""
    if score >= 90:
        return "✅ Excellent"
    elif score >= 70:
        return "🟡 Degraded"
    elif score >= 50:
        return "🟠 Suspect"
    else:
        return "🔴 Compromised"
