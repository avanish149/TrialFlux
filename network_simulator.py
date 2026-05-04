"""
network_simulator.py — Unreliable Network Transmission Simulator
-----------------------------------------------------------------
Simulates real-world hostile network conditions on a stream of
CompressedSegment objects:

  • Packet loss        — segments randomly dropped
  • Bit corruption     — random noise injected into coefficients
  • Delay / jitter     — (conceptual; reflected in ordering metadata)
  • Packet reordering  — segments shuffled out of order
  • Replay attack      — a past segment is duplicated later in the stream
  • Fabricated segment — a synthetic fake segment is injected

Each affected segment is tagged with a SimEvent metadata dict so the
auditor can later classify it correctly.
"""

import copy
import random
import numpy as np
from dataclasses import dataclass, field
from compressor import CompressedSegment


# ═══════════════════════════════════════════════════════════════════════════════
# SimulationConfig — control knobs for each attack/fault type
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimulationConfig:
    """All network-simulation parameters in one place."""
    seed:              int   = 0

    # Probability each segment is DROPPED (0.0 – 1.0)
    loss_rate:         float = 0.05

    # Probability each segment's coefficients are bit-corrupted
    corruption_rate:   float = 0.08

    # Standard deviation of Gaussian noise added during corruption
    corruption_noise:  float = 150.0

    # Fraction of segments that get reordered (0 = disabled)
    reorder_rate:      float = 0.05

    # Number of replay attacks to inject
    n_replay:          int   = 2

    # Number of fully fabricated (fake) segments to inject
    n_fabricated:      int   = 2


# ═══════════════════════════════════════════════════════════════════════════════
# Transmission result
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TransmitResult:
    """Result of one simulated transmission run."""
    sent_count:      int
    received:        list[CompressedSegment]
    dropped_ids:     list[str]
    corrupted_ids:   list[str]
    reordered_ids:   list[str]
    replayed_ids:    list[str]
    fabricated_ids:  list[str]
    sim_events:      dict[str, list[str]]   # segment_id → list of sim events

    def event_summary(self) -> str:
        lines = [
            f"  Sent:       {self.sent_count}",
            f"  Received:   {len(self.received)}",
            f"  Dropped:    {len(self.dropped_ids)}",
            f"  Corrupted:  {len(self.corrupted_ids)}",
            f"  Reordered:  {len(self.reordered_ids)}",
            f"  Replayed:   {len(self.replayed_ids)}",
            f"  Fabricated: {len(self.fabricated_ids)}",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Individual fault injectors
# ═══════════════════════════════════════════════════════════════════════════════

def _corrupt_segment(seg: CompressedSegment, noise_std: float, rng: np.random.Generator) -> CompressedSegment:
    """Inject Gaussian noise into a deep copy of the segment."""
    corrupted = copy.deepcopy(seg)
    noise     = rng.normal(0, noise_std, size=corrupted.coefficients.shape)
    corrupted.coefficients = corrupted.coefficients + noise
    # Intentionally do NOT recompute the hash → hash will fail on validation
    return corrupted


def _make_fabricated_segment(reference: CompressedSegment, idx: int,
                              rng: np.random.Generator) -> CompressedSegment:
    """
    Create a fully synthetic (fake) segment that did not originate from
    the real EEG encoder.  Looks plausible but the hash will be wrong.
    """
    fake = copy.deepcopy(reference)
    fake.segment_id   = f"FAKE_{reference.channel}_{idx:03d}"
    fake.window_id    = -1
    fake.t_start      = reference.t_start + 0.001  # slightly off
    fake.t_end        = reference.t_end   + 0.001
    fake.priority     = "HIGH"
    fake.priority_score = 95.0
    fake.event_types  = ["fabricated"]
    fake.method       = "fabricated"

    # Fill with random-looking data — enough to fool naive amplitude checks
    n = fake.n_stored
    fake.coefficients = rng.normal(0, 50, size=n).astype(np.float32)
    # Hash is purposely invalid (empty string) — auditor will catch it
    fake.segment_hash = ""
    fake.chain_hash   = ""
    return fake


# ═══════════════════════════════════════════════════════════════════════════════
# Main simulator
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_transmission(
    segments: list[CompressedSegment],
    config:   SimulationConfig | None = None,
) -> TransmitResult:
    """
    Pass `segments` through a simulated hostile network.

    Returns a TransmitResult containing the received (possibly mutated) list
    and full metadata about what happened to each segment.
    """
    if config is None:
        config = SimulationConfig()

    rng  = np.random.default_rng(config.seed)
    rnd  = random.Random(config.seed)

    sent_count     = len(segments)
    dropped_ids:   list[str] = []
    corrupted_ids: list[str] = []
    reordered_ids: list[str] = []
    replayed_ids:  list[str] = []
    fabricated_ids:list[str] = []
    sim_events:    dict[str, list[str]] = {}

    stream: list[CompressedSegment] = []

    # ── 1. Packet loss ─────────────────────────────────────────────────────────
    for seg in segments:
        if rnd.random() < config.loss_rate:
            dropped_ids.append(seg.segment_id)
            sim_events[seg.segment_id] = ["DROPPED"]
        else:
            stream.append(copy.deepcopy(seg))

    # ── 2. Bit corruption ──────────────────────────────────────────────────────
    for i, seg in enumerate(stream):
        if rnd.random() < config.corruption_rate:
            stream[i] = _corrupt_segment(seg, config.corruption_noise, rng)
            corrupted_ids.append(seg.segment_id)
            sim_events.setdefault(seg.segment_id, []).append("CORRUPTED")

    # ── 3. Packet reordering ───────────────────────────────────────────────────
    if config.reorder_rate > 0 and len(stream) >= 4:
        n_swap = max(1, int(len(stream) * config.reorder_rate))
        for _ in range(n_swap):
            i = rnd.randint(0, len(stream) - 2)
            j = rnd.randint(i + 1, min(i + 10, len(stream) - 1))
            stream[i], stream[j] = stream[j], stream[i]
            reordered_ids.extend([stream[i].segment_id, stream[j].segment_id])
            sim_events.setdefault(stream[i].segment_id, []).append("REORDERED")
            sim_events.setdefault(stream[j].segment_id, []).append("REORDERED")

    # ── 4. Replay attacks (duplicate a past segment later in stream) ───────────
    if config.n_replay > 0 and len(stream) >= 2:
        replay_sources = rnd.sample(stream[:max(1, len(stream)//3)], min(config.n_replay, len(stream)//3 or 1))
        for src in replay_sources:
            replayed = copy.deepcopy(src)
            # Insert near the end to maximise disruption
            insert_pos = rnd.randint(len(stream)//2, len(stream))
            stream.insert(insert_pos, replayed)
            replayed_ids.append(src.segment_id)
            sim_events.setdefault(src.segment_id, []).append("REPLAYED")

    # ── 5. Fabricated segment injection ────────────────────────────────────────
    if config.n_fabricated > 0 and len(stream) >= 1:
        for idx in range(config.n_fabricated):
            ref  = rnd.choice(stream)
            fake = _make_fabricated_segment(ref, idx, rng)
            insert_pos = rnd.randint(0, len(stream))
            stream.insert(insert_pos, fake)
            fabricated_ids.append(fake.segment_id)
            sim_events[fake.segment_id] = ["FABRICATED"]

    return TransmitResult(
        sent_count     = sent_count,
        received       = stream,
        dropped_ids    = list(set(dropped_ids)),
        corrupted_ids  = list(set(corrupted_ids)),
        reordered_ids  = list(set(reordered_ids)),
        replayed_ids   = list(set(replayed_ids)),
        fabricated_ids = list(set(fabricated_ids)),
        sim_events     = sim_events,
    )


# ─── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader   import generate_synthetic_eeg
    from event_detector import detect_all_channels
    from compressor    import compress_windows
    from integrity     import sign_chain

    df   = generate_synthetic_eeg(duration_s=20, channels=["Fp1", "C3"])
    wins = detect_all_channels(df)
    segs = sign_chain(compress_windows(wins))

    cfg    = SimulationConfig(loss_rate=0.1, corruption_rate=0.1, n_replay=3, n_fabricated=3)
    result = simulate_transmission(segs, cfg)
    print(result.event_summary())
