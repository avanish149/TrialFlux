"""
classifier.py — Explainable Anomaly Classification Engine
----------------------------------------------------------
Reads AuditRecord evidence and applies a transparent decision tree
to classify every segment into one of four outcomes:

  CLASS_CLEAN       — integrity OK, waveform plausible, no anomalies
  CLASS_BIOLOGICAL  — integrity OK, waveform unusual but clinically plausible
  CLASS_CORRUPTION  — hash/chain fail consistent with network damage
  CLASS_TAMPER      — hash/chain fail + replay / fabrication / ordering attack

Each classification comes with:
  • A confidence score (0.0 – 1.0)
  • A human-readable explanation
  • A list of triggered rules
"""

import pandas as pd
from dataclasses import dataclass
from auditor import AuditRecord
from utils   import (CLASS_BIOLOGICAL, CLASS_CLEAN,
                     CLASS_CORRUPTION, CLASS_TAMPER)


# ═══════════════════════════════════════════════════════════════════════════════
# Classification result
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClassificationResult:
    segment_id:    str
    channel:       str
    t_start:       float
    t_end:         float
    verdict:       str        # one of CLASS_* constants
    confidence:    float      # 0.0 – 1.0
    explanation:   str
    rules_fired:   list[str]
    priority:      str        # from the original segment
    priority_score:float


# ═══════════════════════════════════════════════════════════════════════════════
# Decision rules
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_one(rec: AuditRecord, priority: str, priority_score: float) -> ClassificationResult:
    """
    Apply an explicit, ordered rule-set to produce a classification.

    Rule priority (highest first):
      R1. Fabricated tag           → TAMPER  (strongest signal)
      R2. Replay / duplicate       → TAMPER
      R3. Ordering violation + hash fail → TAMPER
      R4. Hash fail + chain fail   → CORRUPTION (network damage)
      R5. Chain fail only          → CORRUPTION (reorder / loss)
      R6. Hash fail only           → CORRUPTION (partial corruption)
      R7. All integrity OK + high event score → BIOLOGICAL EVENT
      R8. All integrity OK + no event → CLEAN
    """
    rules_fired: list[str] = []
    ev = set(rec.evidence)

    # ── R1: Fabrication ────────────────────────────────────────────────────────
    if rec.is_fabricated or "fabricated_tag" in ev:
        rules_fired.append("R1:fabricated_tag")
        return ClassificationResult(
            segment_id    = rec.segment_id,
            channel       = rec.channel,
            t_start       = rec.t_start,
            t_end         = rec.t_end,
            verdict       = CLASS_TAMPER,
            confidence    = 0.97,
            explanation   = (
                "Segment carries a fabrication marker (method='fabricated' or "
                "FAKE_ prefix). This segment did not originate from a genuine "
                "EEG encoder and was injected into the stream."
            ),
            rules_fired   = rules_fired,
            priority      = priority,
            priority_score= priority_score,
        )

    # ── R2: Replay / duplication ────────────────────────────────────────────────
    if rec.is_replay:
        rules_fired.append("R2:replay_duplicate")
        return ClassificationResult(
            segment_id    = rec.segment_id,
            channel       = rec.channel,
            t_start       = rec.t_start,
            t_end         = rec.t_end,
            verdict       = CLASS_TAMPER,
            confidence    = 0.93,
            explanation   = (
                "Segment ID was already received earlier in the stream. "
                "Replay attack detected: an adversary re-transmitted a "
                "previously captured packet."
            ),
            rules_fired   = rules_fired,
            priority      = priority,
            priority_score= priority_score,
        )

    # ── R3: Ordering violation + integrity failure → coordinated attack ─────────
    if "ordering_violation" in ev and (not rec.hash_ok or not rec.chain_ok):
        rules_fired.append("R3:ordering+integrity_fail")
        return ClassificationResult(
            segment_id    = rec.segment_id,
            channel       = rec.channel,
            t_start       = rec.t_start,
            t_end         = rec.t_end,
            verdict       = CLASS_TAMPER,
            confidence    = 0.88,
            explanation   = (
                "Segment is out of expected order AND fails integrity checks. "
                "This combination is characteristic of an active man-in-the-middle "
                "attack or deliberate packet reordering with content modification."
            ),
            rules_fired   = rules_fired,
            priority      = priority,
            priority_score= priority_score,
        )

    # ── R4: Both hash and chain fail → severe network corruption or tampering ───
    if not rec.hash_ok and not rec.chain_ok:
        rules_fired.append("R4:hash_and_chain_fail")
        # Distinguish corruption vs tamper by waveform plausibility
        waveform_ok = rec.amplitude_ok and rec.rms_ok and rec.spectral_ok
        if not waveform_ok:
            # Waveform also broken → more consistent with random corruption
            rules_fired.append("R4a:waveform_implausible→corruption")
            return ClassificationResult(
                segment_id    = rec.segment_id,
                channel       = rec.channel,
                t_start       = rec.t_start,
                t_end         = rec.t_end,
                verdict       = CLASS_CORRUPTION,
                confidence    = 0.85,
                explanation   = (
                    "Both segment hash and chain hash failed, and the reconstructed "
                    "waveform is physiologically implausible. Consistent with severe "
                    "packet corruption during transmission."
                ),
                rules_fired   = rules_fired,
                priority      = priority,
                priority_score= priority_score,
            )
        else:
            # Waveform looks OK but hashes fail → suspicious, leaning tamper
            rules_fired.append("R4b:waveform_ok→possible_tamper")
            return ClassificationResult(
                segment_id    = rec.segment_id,
                channel       = rec.channel,
                t_start       = rec.t_start,
                t_end         = rec.t_end,
                verdict       = CLASS_TAMPER,
                confidence    = 0.78,
                explanation   = (
                    "Hash and chain both fail, but the waveform looks plausible. "
                    "A careful attacker may have crafted a realistic-looking segment "
                    "with an invalid cryptographic signature."
                ),
                rules_fired   = rules_fired,
                priority      = priority,
                priority_score= priority_score,
            )

    # ── R5: Chain fail only (segment content OK, but chain is broken) ───────────
    if rec.hash_ok and not rec.chain_ok:
        rules_fired.append("R5:chain_fail_only")
        return ClassificationResult(
            segment_id    = rec.segment_id,
            channel       = rec.channel,
            t_start       = rec.t_start,
            t_end         = rec.t_end,
            verdict       = CLASS_CORRUPTION,
            confidence    = 0.80,
            explanation   = (
                "Segment content hash is valid but the chain hash is broken. "
                "A preceding segment was likely dropped or reordered, snapping "
                "the provenance chain — consistent with packet loss or reordering."
            ),
            rules_fired   = rules_fired,
            priority      = priority,
            priority_score= priority_score,
        )

    # ── R6: Hash fail only ──────────────────────────────────────────────────────
    if not rec.hash_ok:
        rules_fired.append("R6:hash_fail_only")
        return ClassificationResult(
            segment_id    = rec.segment_id,
            channel       = rec.channel,
            t_start       = rec.t_start,
            t_end         = rec.t_end,
            verdict       = CLASS_CORRUPTION,
            confidence    = 0.75,
            explanation   = (
                "Segment hash does not match recorded value. Content was altered "
                "in transit — most likely bit-flip corruption or partial packet loss."
            ),
            rules_fired   = rules_fired,
            priority      = priority,
            priority_score= priority_score,
        )

    # ── Integrity clean from here ───────────────────────────────────────────────

    # ── R7: Integrity OK + high-priority event → biological ────────────────────
    if priority == "HIGH" and priority_score >= 60:
        rules_fired.append("R7:integrity_ok+high_priority_event")
        conf = min(0.95, 0.70 + priority_score / 500)
        return ClassificationResult(
            segment_id    = rec.segment_id,
            channel       = rec.channel,
            t_start       = rec.t_start,
            t_end         = rec.t_end,
            verdict       = CLASS_BIOLOGICAL,
            confidence    = round(conf, 3),
            explanation   = (
                f"All integrity checks pass and the segment has a high clinical "
                f"priority score ({priority_score:.1f}/100). The unusual waveform "
                f"pattern is consistent with a genuine brain event (seizure, spike, "
                f"or burst) rather than transmission artefact."
            ),
            rules_fired   = rules_fired,
            priority      = priority,
            priority_score= priority_score,
        )

    # ── R8: Clean ───────────────────────────────────────────────────────────────
    rules_fired.append("R8:all_clear")
    return ClassificationResult(
        segment_id    = rec.segment_id,
        channel       = rec.channel,
        t_start       = rec.t_start,
        t_end         = rec.t_end,
        verdict       = CLASS_CLEAN,
        confidence    = 0.92,
        explanation   = (
            "Segment passed all integrity checks, waveform is physiologically "
            "plausible, and no anomaly score threshold was exceeded. "
            "Data appears clean and trustworthy."
        ),
        rules_fired   = rules_fired,
        priority      = priority,
        priority_score= priority_score,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Batch classifier
# ═══════════════════════════════════════════════════════════════════════════════

def classify_all(
    audit_records: list[AuditRecord],
    segments:      list["CompressedSegment"],  # for priority metadata
) -> list[ClassificationResult]:
    """
    Classify every audit record.  Returns one ClassificationResult per record.
    """
    seg_meta = {s.segment_id: (s.priority, s.priority_score) for s in segments}
    results: list[ClassificationResult] = []

    for rec in audit_records:
        prio, pscore = seg_meta.get(rec.segment_id, ("LOW", 0.0))
        results.append(_classify_one(rec, prio, pscore))

    return results


def classification_summary(results: list[ClassificationResult]) -> pd.DataFrame:
    """Return a tidy DataFrame with all classification results."""
    rows = []
    for r in results:
        rows.append({
            "segment_id":    r.segment_id,
            "channel":       r.channel,
            "t_start":       r.t_start,
            "t_end":         r.t_end,
            "verdict":       r.verdict,
            "confidence":    r.confidence,
            "priority":      r.priority,
            "priority_score":r.priority_score,
            "rules":         " | ".join(r.rules_fired),
            "explanation":   r.explanation,
        })
    return pd.DataFrame(rows)


# ─── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader        import generate_synthetic_eeg
    from event_detector     import detect_all_channels
    from compressor         import compress_windows
    from integrity          import sign_chain, validate_chain, extract_reference_hashes
    from network_simulator  import simulate_transmission, SimulationConfig
    from auditor            import audit_segments

    df   = generate_synthetic_eeg(duration_s=30, channels=["Fp1", "C3"])
    segs = sign_chain(compress_windows(detect_all_channels(df)))
    ref  = extract_reference_hashes(segs)

    cfg    = SimulationConfig(corruption_rate=0.12, n_fabricated=3, n_replay=2)
    result = simulate_transmission(segs, cfg)
    verd   = validate_chain(result.received, ref)
    recs   = audit_segments(result.received, verd, result.sim_events)
    cls    = classify_all(recs, result.received)
    df_out = classification_summary(cls)

    print(df_out[["channel","t_start","verdict","confidence","rules"]].to_string(index=False))
    print("\nVerdict counts:")
    print(df_out["verdict"].value_counts().to_string())
