"""
integrity.py — Merkle-Like Hash Chain for EEG Segment Provenance
-----------------------------------------------------------------
Each compressed segment receives:
  1. A segment hash   = SHA-256( segment_id | coefficients | metadata )
  2. A chain hash     = SHA-256( prev_chain_hash | segment_hash )

This creates a tamper-evident chain: modifying any segment breaks
all subsequent chain hashes, and modifying the order breaks the chain too.

Validation returns per-segment verdicts:
  VALID      — hash and chain match
  HASH_FAIL  — segment content was modified
  CHAIN_FAIL — segment is out of order or chain was broken
  MISSING    — segment is absent (gap detected)
"""

import hashlib
import json
import numpy as np
from dataclasses import dataclass
from compressor import CompressedSegment


# ─── Genesis hash (chain root) ──────────────────────────────────────────────────
GENESIS_HASH = "0" * 64


# ═══════════════════════════════════════════════════════════════════════════════
# Hashing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _bytes_of(seg: CompressedSegment) -> bytes:
    """
    Produce a deterministic byte representation of a segment's content
    for hashing purposes.  We serialise the coefficients as a fixed-point
    hex string to avoid float representation drift.
    """
    coeff_hex = seg.coefficients.astype(np.float32).tobytes().hex()
    meta = json.dumps({
        "segment_id":    seg.segment_id,
        "channel":       seg.channel,
        "t_start":       round(seg.t_start, 6),
        "t_end":         round(seg.t_end, 6),
        "priority":      seg.priority,
        "n_original":    seg.n_original,
        "n_stored":      seg.n_stored,
        "method":        seg.method,
    }, sort_keys=True)
    return (meta + coeff_hex).encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# Build chain
# ═══════════════════════════════════════════════════════════════════════════════

def sign_chain(segments: list[CompressedSegment]) -> list[CompressedSegment]:
    """
    Compute and attach segment_hash and chain_hash to every segment in-place.
    Segments are expected to be sorted by (channel, window_id).
    Returns the same list (mutated).
    """
    prev_chain = GENESIS_HASH
    for seg in segments:
        seg.segment_hash = _sha256(_bytes_of(seg))
        seg.chain_hash   = _sha256((prev_chain + seg.segment_hash).encode())
        prev_chain       = seg.chain_hash
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# Validation result
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SegmentVerdict:
    segment_id:   str
    channel:      str
    window_id:    int
    t_start:      float
    t_end:        float
    hash_ok:      bool
    chain_ok:     bool
    verdict:      str       # VALID | HASH_FAIL | CHAIN_FAIL | MISSING
    stored_hash:  str
    computed_hash:str
    stored_chain: str
    computed_chain:str


def validate_chain(
    segments: list[CompressedSegment],
    reference_hashes: dict[str, str] | None = None,
) -> list[SegmentVerdict]:
    """
    Validate the hash chain of a (possibly corrupted) segment list.

    Parameters
    ----------
    segments          : received segment list (may have gaps / modifications)
    reference_hashes  : optional dict mapping segment_id → original segment_hash
                        (simulates a trusted manifest from the sender).
                        When None, we only check internal chain consistency.

    Returns a list of SegmentVerdict, one per segment.
    """
    verdicts: list[SegmentVerdict] = []
    prev_chain = GENESIS_HASH

    for seg in segments:
        computed_seg_hash   = _sha256(_bytes_of(seg))
        computed_chain_hash = _sha256((prev_chain + computed_seg_hash).encode())

        hash_ok  = computed_seg_hash == seg.segment_hash
        chain_ok = computed_chain_hash == seg.chain_hash

        # If a reference manifest is provided, also cross-check
        if reference_hashes and seg.segment_id in reference_hashes:
            ref_hash = reference_hashes[seg.segment_id]
            hash_ok  = hash_ok and (computed_seg_hash == ref_hash)

        if hash_ok and chain_ok:
            verdict = "VALID"
        elif not hash_ok and not chain_ok:
            verdict = "HASH_FAIL"
        elif hash_ok and not chain_ok:
            verdict = "CHAIN_FAIL"
        else:
            verdict = "HASH_FAIL"     # chain_ok but hash wrong → data drift

        verdicts.append(SegmentVerdict(
            segment_id    = seg.segment_id,
            channel       = seg.channel,
            window_id     = seg.window_id,
            t_start       = seg.t_start,
            t_end         = seg.t_end,
            hash_ok       = hash_ok,
            chain_ok      = chain_ok,
            verdict       = verdict,
            stored_hash   = seg.segment_hash[:12] + "…",
            computed_hash = computed_seg_hash[:12] + "…",
            stored_chain  = seg.chain_hash[:12] + "…",
            computed_chain= computed_chain_hash[:12] + "…",
        ))
        # Advance chain with the (possibly corrupt) stored chain hash
        # so we can detect further chain breaks downstream
        prev_chain = seg.chain_hash if seg.chain_hash else computed_chain_hash

    return verdicts


def integrity_score(verdicts: list[SegmentVerdict]) -> float:
    """
    Return an overall integrity score 0-100.
    Score = (valid segments / total) × 100,
    with extra penalty for CHAIN_FAIL (implies ordering attack).
    """
    if not verdicts:
        return 0.0
    valid   = sum(1 for v in verdicts if v.verdict == "VALID")
    chain_f = sum(1 for v in verdicts if v.verdict == "CHAIN_FAIL")
    base    = valid / len(verdicts) * 100
    penalty = chain_f / len(verdicts) * 20   # chain breaks are extra bad
    return max(0.0, round(base - penalty, 2))


# ─── Reference manifest extraction ─────────────────────────────────────────────

def extract_reference_hashes(segments: list[CompressedSegment]) -> dict[str, str]:
    """Extract {segment_id: segment_hash} from a signed segment list."""
    return {seg.segment_id: seg.segment_hash for seg in segments}


# ─── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import generate_synthetic_eeg
    from event_detector import detect_all_channels
    from compressor import compress_windows

    df   = generate_synthetic_eeg(duration_s=10, channels=["Fp1"])
    wins = detect_all_channels(df)
    segs = compress_windows(wins)
    segs = sign_chain(segs)

    ref  = extract_reference_hashes(segs)

    # Tamper with one segment
    segs[3].coefficients[0] += 9999.0
    segs[3].segment_hash    = "deadbeef" * 8

    verdicts = validate_chain(segs, ref)
    for v in verdicts[:6]:
        print(f"{v.segment_id}: {v.verdict}  hash_ok={v.hash_ok} chain_ok={v.chain_ok}")

    print(f"\nIntegrity score: {integrity_score(verdicts):.1f} / 100")
