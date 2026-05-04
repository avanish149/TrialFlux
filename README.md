# TrialFlux 🧠
### Smart Brain Signal Transmission & Integrity Checker
> *Biomedical Hackathon Prototype — End-to-End EEG Provenance Pipeline*

---

## What It Does

TrialFlux is a **provenance-aware EEG telemetry pipeline** that answers the critical question:

> *"Is this unusual EEG pattern a real brain event, a network error, or tampered data?"*

It does so through 6 automated stages:

| Stage | Module | What Happens |
|-------|--------|--------------|
| 1 | `data_loader.py` | Load CSV **or** generate synthetic multi-channel EEG with seizures/spikes |
| 2 | `event_detector.py` | Sliding-window analysis → HIGH / LOW priority with 6 clinical heuristics |
| 3 | `compressor.py` | DCT-based adaptive compression (lossless for HIGH, lossy for LOW priority) |
| 4 | `integrity.py` | SHA-256 segment hashes + Merkle-like chain linking |
| 5 | `network_simulator.py` | Simulate packet loss, bit corruption, replay attacks, fabricated segments |
| 6 | `auditor.py` + `classifier.py` | Per-segment audit + explainable rule-based classification |
| — | `app.py` | Streamlit dashboard with 6 interactive tabs |

---

## Quick Start

```bash
# 1. Clone / navigate to the project folder
cd TrialFlux

# 2. Install dependencies (Python 3.11+ recommended)
pip install -r requirements.txt

# 3. Generate the sample EEG CSV (optional — app auto-generates if missing)
python data_loader.py

# 4. Launch the dashboard
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Project Structure

```
TrialFlux/
├── app.py                 ← Streamlit dashboard (6 tabs)
├── data_loader.py         ← EEG ingestion + synthetic generator
├── event_detector.py      ← Sliding-window event detection
├── compressor.py          ← Adaptive DCT compression
├── integrity.py           ← Merkle-like hash chain
├── network_simulator.py   ← Fault & attack injection
├── auditor.py             ← Per-segment evidence collector
├── classifier.py          ← Explainable 8-rule verdict engine
├── utils.py               ← Shared constants & helpers
├── requirements.txt
└── sample_data/
    └── eeg_sample.csv     ← Auto-generated on first run
```

---

## Classification Logic (Explainable Rules)

```
R1  fabricated_tag          → Tampering / Fabrication  (conf 0.97)
R2  replay_duplicate        → Tampering / Fabrication  (conf 0.93)
R3  ordering + hash fail    → Tampering / Fabrication  (conf 0.88)
R4a hash+chain fail, bad waveform → Network Corruption (conf 0.85)
R4b hash+chain fail, good waveform→ Tampering          (conf 0.78)
R5  chain fail only         → Network Corruption       (conf 0.80)
R6  hash fail only          → Network Corruption       (conf 0.75)
R7  integrity OK + HIGH score → Biological Event       (conf 0.70–0.95)
R8  all clear               → Clean                    (conf 0.92)
```

---

## Dashboard Tabs

| Tab | Content |
|-----|---------|
| 📈 EEG Signals | Original vs reconstructed vs residual, SNR |
| 🔍 Event Timeline | Priority bubble chart, event distribution |
| 🌐 Transmission | Network fault breakdown bar chart |
| 🔐 Integrity Chain | Hash/chain verdict table + pie chart |
| 🧩 Classification | Verdict donut + confidence scatter + full table |
| 📋 Audit Log | Filterable per-segment evidence log, CSV export |

---

## Running Module Self-Tests

```bash
python data_loader.py        # generates sample CSV
python event_detector.py     # prints HIGH-priority windows
python compressor.py         # prints compression ratios
python integrity.py          # shows hash chain with a tamper injection
python network_simulator.py  # shows transmission event summary
python auditor.py            # per-segment audit records
python classifier.py         # verdict counts
```

---

## Tech Stack

- **Python 3.11+**
- **NumPy / SciPy** — signal processing, DCT, Welch PSD
- **Pandas** — data wrangling
- **Plotly** — interactive charts
- **Streamlit** — dashboard framework
- **hashlib** — SHA-256 integrity

---

*TrialFlux · Biomedical Hackathon 2024 · Software-only prototype*
