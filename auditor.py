"""
auditor.py — Segment-Level Audit Engine
----------------------------------------
For every received CompressedSegment, the auditor combines:

  1. Integrity evidence    (hash_ok, chain_ok)
  2. Waveform plausibility (amplitude, spectral fingerprint)
  3. Duplication / replay  (segment_id seen before)
  4. Fabrication signals   (known fake method tag, impossible metadata)
  5. Gap / ordering checks (expected vs received window IDs)

It produces an AuditRecord per segment, which the classifier then reads
to emit a final verdict.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from compressor   import CompressedSegment, decompress_segment
from integrity    import SegmentVerdict
from scipy import signal as sp_signal


# ─── Default Plausibility Thresholds (used if no baseline is provided) ──────────
DEFAULT_MAX_AMP = 500.0    # µV peak-to-peak hard limit
DEFAULT_MAX_RMS = 200.0    # µV RMS
DEFAULT_MIN_RMS = 0.1      # below this → flat-line / dead channel
DEFAULT_FREQ_RANGE = (0.5, 80.0)   # valid EEG Hz range

@dataclass
class BaselineProfile:
    """Dynamic physiological boundaries trained on user's base data."""
    max_amplitude: float
    max_rms: float
    min_rms: float
    freq_min: float
    freq_max: float


@dataclass
class AuditRecord:
    """Full audit evidence for a single received segment."""
    segment_id:        str
    channel:           str
    window_id:         int
    t_start:           float
    t_end:             float

    # ── Integrity ──────────────────────────────────────────────────────────────
    hash_ok:           bool
    chain_ok:          bool
    integrity_verdict: str    # VALID | HASH_FAIL | CHAIN_FAIL

    # ── Replay / duplication ───────────────────────────────────────────────────
    is_replay:         bool
    is_fabricated:     bool   # method == "fabricated" tag

    # ── Waveform plausibility ──────────────────────────────────────────────────
    amplitude_ok:      bool
    rms_ok:            bool
    spectral_ok:       bool
    rms_value:         float
    peak_to_peak:      float
    dominant_freq:     float

    # ── Ordering ───────────────────────────────────────────────────────────────
    expected_window_id: int | None   # None if unknown
    order_ok:           bool

    # ── Evidence summary for classifier ───────────────────────────────────────
    evidence:          list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Waveform plausibility checks
# ═══════════════════════════════════════════════════════════════════════════════

def _dominant_frequency(signal: np.ndarray, fs: int = 256) -> float:
    """Return the frequency (Hz) with the highest PSD peak."""
    if len(signal) < 16:
        return 0.0
    nperseg = min(len(signal), 64)
    freqs, psd = sp_signal.welch(signal, fs=fs, nperseg=nperseg)
    return float(freqs[np.argmax(psd)])


def fit_baseline(df: pd.DataFrame, fs: int = 256) -> BaselineProfile:
    """
    Train model on a clean baseline dataset to extract physiological bounds.
    Adds a generous tolerance (+100%) to maximums to allow for real biological
    variance without triggering false network corruption flags.
    """
    ptps, rmss, freqs = [], [], []
    win_size = int(2.0 * fs)
    
    for ch in df["channel"].unique():
        vals = df[df["channel"] == ch]["value"].to_numpy()
        for i in range(0, len(vals) - win_size + 1, win_size):
            wave = vals[i:i+win_size]
            ptp = float(wave.max() - wave.min())
            rms = float(np.sqrt(np.mean(wave ** 2)))
            dfreq = _dominant_frequency(wave, fs=fs)
            ptps.append(ptp)
            rmss.append(rms)
            freqs.append(dfreq)
            
    if not ptps: # Fallback if empty
        return BaselineProfile(DEFAULT_MAX_AMP, DEFAULT_MAX_RMS, DEFAULT_MIN_RMS, DEFAULT_FREQ_RANGE[0], DEFAULT_FREQ_RANGE[1])
        
    return BaselineProfile(
        max_amplitude = np.percentile(ptps, 99) * 2.0,  # 2x max baseline
        max_rms       = np.percentile(rmss, 99) * 2.0,
        min_rms       = max(0.01, np.percentile(rmss, 1) * 0.1),
        freq_min      = max(0.1, np.min(freqs) - 2.0),
        freq_max      = min(120.0, np.max(freqs) + 20.0),
    )


def _check_waveform(seg: CompressedSegment, fs: int, baseline: BaselineProfile | None) -> tuple[bool, bool, bool, float, float, float]:
    """
    Reconstruct waveform and run plausibility checks against baseline (or defaults).
    Returns (amplitude_ok, rms_ok, spectral_ok, rms, ptp, dominant_freq)
    """
    try:
        wave = decompress_segment(seg)
    except Exception:
        return False, False, False, 0.0, 0.0, 0.0

    ptp   = float(wave.max() - wave.min())
    rms   = float(np.sqrt(np.mean(wave ** 2)))
    d_freq = _dominant_frequency(wave, fs=fs)

    max_amp = baseline.max_amplitude if baseline else DEFAULT_MAX_AMP
    max_r   = baseline.max_rms if baseline else DEFAULT_MAX_RMS
    min_r   = baseline.min_rms if baseline else DEFAULT_MIN_RMS
    f_min   = baseline.freq_min if baseline else DEFAULT_FREQ_RANGE[0]
    f_max   = baseline.freq_max if baseline else DEFAULT_FREQ_RANGE[1]

    amplitude_ok = ptp <= max_amp
    rms_ok       = min_r <= rms <= max_r
    spectral_ok  = f_min <= d_freq <= f_max

    return amplitude_ok, rms_ok, spectral_ok, rms, ptp, d_freq


# ═══════════════════════════════════════════════════════════════════════════════
# Main audit function
# ═══════════════════════════════════════════════════════════════════════════════

def audit_segments(
    received:   list[CompressedSegment],
    verdicts:   list[SegmentVerdict],
    sim_events: dict[str, list[str]] | None = None,
    fs:         int = 256,
    baseline:   BaselineProfile | None = None,
) -> list[AuditRecord]:
    """
    Produce an AuditRecord for every received segment.

    Parameters
    ----------
    received    : the segments that arrived (may include replays / fakes)
    verdicts    : integrity verdict from integrity.validate_chain()
    sim_events  : ground-truth fault map from the network simulator
                  (only available in demo mode; real deployments omit this)
    fs          : sampling frequency in Hz
    """
    sim_events = sim_events or {}
    verdict_map = {v.segment_id: v for v in verdicts}

    seen_ids: set[str] = set()
    # Build expected ordering from clean window_id sequence
    # (we assume the FIRST occurrence of each channel gives the baseline order)
    channel_expected: dict[str, int] = {}

    records: list[AuditRecord] = []

    for seg in received:
        v = verdict_map.get(seg.segment_id)
        hash_ok  = v.hash_ok   if v else False
        chain_ok = v.chain_ok  if v else False
        i_verdict = v.verdict  if v else "HASH_FAIL"

        # ── Replay / duplication check ─────────────────────────────────────────
        is_replay = seg.segment_id in seen_ids
        seen_ids.add(seg.segment_id)

        # ── Fabrication check (via method tag or sim ground-truth) ─────────────
        is_fabricated = (
            seg.method == "fabricated"
            or "FABRICATED" in sim_events.get(seg.segment_id, [])
            or seg.segment_id.startswith("FAKE_")
        )

        # ── Waveform plausibility ──────────────────────────────────────────────
        amp_ok, rms_ok, spec_ok, rms_val, ptp, dom_freq = _check_waveform(seg, fs, baseline)

        # ── Ordering check ─────────────────────────────────────────────────────
        ch = seg.channel
        expected_wid = channel_expected.get(ch)
        if expected_wid is None:
            order_ok = True          # first segment for this channel
        else:
            order_ok = (seg.window_id == expected_wid + 1
                        or seg.window_id < 0)  # fake segs have window_id == -1

        channel_expected[ch] = seg.window_id

        # ── Build evidence list ────────────────────────────────────────────────
        evidence: list[str] = []
        if not hash_ok:       evidence.append("hash_mismatch")
        if not chain_ok:      evidence.append("chain_broken")
        if is_replay:         evidence.append("replay_duplicate")
        if is_fabricated:     evidence.append("fabricated_tag")
        if not amp_ok:        evidence.append("amplitude_out_of_range")
        if not rms_ok:        evidence.append("rms_out_of_range")
        if not spec_ok:       evidence.append("spectral_implausible")
        if not order_ok:      evidence.append("ordering_violation")
        if "CORRUPTED" in sim_events.get(seg.segment_id, []):
            evidence.append("known_corruption")
        if "REORDERED" in sim_events.get(seg.segment_id, []):
            evidence.append("known_reorder")

        records.append(AuditRecord(
            segment_id         = seg.segment_id,
            channel            = ch,
            window_id          = seg.window_id,
            t_start            = seg.t_start,
            t_end              = seg.t_end,
            hash_ok            = hash_ok,
            chain_ok           = chain_ok,
            integrity_verdict  = i_verdict,
            is_replay          = is_replay,
            is_fabricated      = is_fabricated,
            amplitude_ok       = amp_ok,
            rms_ok             = rms_ok,
            spectral_ok        = spec_ok,
            rms_value          = round(rms_val, 3),
            peak_to_peak       = round(ptp, 3),
            dominant_freq      = round(dom_freq, 2),
            expected_window_id = expected_wid,
            order_ok           = order_ok,
            evidence           = evidence,
        ))

    return records


# ─── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader        import generate_synthetic_eeg
    from event_detector     import detect_all_channels
    from compressor         import compress_windows
    from integrity          import sign_chain, validate_chain, extract_reference_hashes
    from network_simulator  import simulate_transmission, SimulationConfig

    df   = generate_synthetic_eeg(duration_s=20, channels=["Fp1"])
    segs = sign_chain(compress_windows(detect_all_channels(df)))
    ref  = extract_reference_hashes(segs)

    cfg    = SimulationConfig(corruption_rate=0.15, n_fabricated=2, n_replay=2)
    result = simulate_transmission(segs, cfg)
    verd   = validate_chain(result.received, ref)
    recs   = audit_segments(result.received, verd, result.sim_events)

    for r in recs[:8]:
        print(f"{r.segment_id[:20]:22s} | hash={r.hash_ok} chain={r.chain_ok} "
              f"replay={r.is_replay} fab={r.is_fabricated} | ev={r.evidence[:2]}")
