"""
app.py — TrialFlux Streamlit Dashboard
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data_loader       import generate_synthetic_eeg, load_eeg_csv, write_sample_csv
from event_detector    import detect_all_channels, summarise_events
from compressor        import compress_windows, decompress_all, build_manifest
from integrity         import sign_chain, validate_chain, extract_reference_hashes, integrity_score
from network_simulator import simulate_transmission, SimulationConfig
from auditor           import audit_segments, fit_baseline, BaselineProfile, fit_baseline, BaselineProfile
from classifier        import classify_all, classification_summary
from utils             import COLORS, VERDICT_COLOR, integrity_badge, snr_db
from utils             import CLASS_BIOLOGICAL, CLASS_CLEAN, CLASS_CORRUPTION, CLASS_TAMPER

# ─── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrialFlux — Neural Telemetry Integrity Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0F1117; color: #E2E8F0; }

.metric-card {
    background: linear-gradient(135deg, #1A1D27 0%, #1e2235 100%);
    border: 1px solid #2D3748;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.metric-card h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
.metric-card p  { color: #8892A4; font-size: 0.8rem; margin: 0.2rem 0 0; text-transform: uppercase; letter-spacing: 0.08em; }

.verdict-pill {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}

.section-header {
    font-size: 1.1rem; font-weight: 600;
    color: #6C63FF;
    border-left: 3px solid #6C63FF;
    padding-left: 0.7rem;
    margin: 1.2rem 0 0.8rem;
}

.stAlert { border-radius: 10px; }
.stTabs [data-baseweb="tab"] { font-weight: 500; }

div[data-testid="stSidebarContent"] {
    background: linear-gradient(180deg, #0F1117 0%, #13161f 100%);
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

VERDICT_EMOJI = {
    CLASS_CLEAN:      "✅",
    CLASS_BIOLOGICAL: "🧠",
    CLASS_CORRUPTION: "⚠️",
    CLASS_TAMPER:     "🔴",
}

def color_verdict(v):
    colors = {CLASS_CLEAN:"#36D399", CLASS_BIOLOGICAL:"#00D4AA",
               CLASS_CORRUPTION:"#FFC947", CLASS_TAMPER:"#FF4E5B"}
    return f"color:{colors.get(v,'#fff')};font-weight:600"


def metric_card(label, value, color="#6C63FF", sub=""):
    st.markdown(f"""
    <div class="metric-card">
      <h1 style="color:{color}">{value}</h1>
      <p>{label}</p>
      {"<small style='color:#8892A4'>"+sub+"</small>" if sub else ""}
    </div>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def run_pipeline(duration, n_channels, loss, corruption, n_replay, n_fab, seed, fs=256, baseline=None):
    """
    Core pipeline — all stages wired together.
    `fs` is the true sampling frequency; defaults to 256 Hz for synthetic data
    but is set from the real CSV when a file is uploaded.
    """
    from utils import EEG_CHANNELS
    channels = EEG_CHANNELS[:n_channels]

    df       = generate_synthetic_eeg(duration_s=duration, channels=channels, seed=seed)
    windows  = detect_all_channels(df, fs=fs)
    segs     = sign_chain(compress_windows(windows))
    ref      = extract_reference_hashes(segs)
    manifest = build_manifest(segs)

    cfg      = SimulationConfig(
        loss_rate=loss, corruption_rate=corruption,
        n_replay=n_replay, n_fabricated=n_fab, seed=seed
    )
    tx       = simulate_transmission(segs, cfg)
    verdicts = validate_chain(tx.received, ref)
    iscore   = integrity_score(verdicts)
    recs     = audit_segments(tx.received, verdicts, tx.sim_events, fs=fs, baseline=baseline)
    cls_res  = classify_all(recs, tx.received)
    cls_df   = classification_summary(cls_res)

    recon    = decompress_all(tx.received)
    ev_df    = summarise_events(windows)

    return df, windows, segs, ref, manifest, tx, verdicts, iscore, recs, cls_res, cls_df, recon, ev_df, fs


@st.cache_data(show_spinner=False)
def run_pipeline_from_df(df_json, fs, loss, corruption, n_replay, n_fab, seed, baseline=None):
    """
    Pipeline variant that starts from a pre-loaded DataFrame
    (used when the user uploads a real CSV).
    `df_json` is the serialised DataFrame (JSON string for cache key compatibility).
    """
    import io
    df      = pd.read_json(io.StringIO(df_json))
    windows = detect_all_channels(df, fs=fs)
    segs    = sign_chain(compress_windows(windows))
    ref     = extract_reference_hashes(segs)
    manifest= build_manifest(segs)
    cfg     = SimulationConfig(loss_rate=loss, corruption_rate=corruption,
                               n_replay=n_replay, n_fabricated=n_fab, seed=seed)
    tx      = simulate_transmission(segs, cfg)
    verdicts= validate_chain(tx.received, ref)
    iscore  = integrity_score(verdicts)
    recs    = audit_segments(tx.received, verdicts, tx.sim_events, fs=fs, baseline=baseline)
    cls_res = classify_all(recs, tx.received)
    cls_df  = classification_summary(cls_res)
    recon   = decompress_all(tx.received)
    ev_df   = summarise_events(windows)
    return df, windows, segs, ref, manifest, tx, verdicts, iscore, recs, cls_res, cls_df, recon, ev_df, fs


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🧠 TrialFlux")
    st.caption("Neural Telemetry Integrity Engine")
    st.divider()

    st.markdown("**📡 EEG Signal**")
    duration   = st.slider("Duration (seconds)", 10, 60, 30)
    n_channels = st.slider("Channels", 1, 8, 4)
    seed       = st.number_input("Random seed", 0, 999, 42)

    st.divider()
    st.markdown("**🌐 Network Simulation**")
    loss       = st.slider("Packet loss rate",      0.0, 0.5, 0.05, 0.01)
    corruption = st.slider("Bit corruption rate",   0.0, 0.5, 0.08, 0.01)
    n_replay   = st.slider("Replay attacks",        0, 10, 2)
    n_fab      = st.slider("Fabricated segments",   0, 10, 2)

    st.divider()
    st.markdown("**\U0001f39b\ufe0f Step 1: Train Baseline**")
    st.caption("Upload clean base data so the model can learn your normal physiological ranges.")
    upload_base = st.file_uploader("Upload Base CSV", type="csv", key="base")

    st.divider()
    st.markdown("**\U0001f4c2 Step 2: Test Data Stream**")
    st.caption("Upload a new file to simulate live network reception and classify anomalies.")
    upload_test = st.file_uploader("Upload Test CSV", type="csv", key="test")
    fs_hint     = st.number_input("Sampling rate override (Hz, 0=auto)", 0, 10000, 0)
    
    st.divider()
    run_btn = st.button("\u25b6 Run Simulation", use_container_width=True, type="primary")

# ═══════════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="background:linear-gradient(135deg,#1a1d27,#1e2235);border-radius:16px;
            padding:2rem 2.5rem;border:1px solid #2D3748;margin-bottom:1.5rem">
  <h1 style="margin:0;font-size:2rem;font-weight:700;
             background:linear-gradient(90deg,#6C63FF,#00D4AA);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent">
    TrialFlux &nbsp;🧠
  </h1>
  <p style="color:#8892A4;margin:0.4rem 0 0;font-size:1rem">
    Smart Brain Signal Transmission &amp; Integrity Checker
    &nbsp;·&nbsp; End-to-End EEG Provenance Pipeline
  </p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Load & run
# ═══════════════════════════════════════════════════════════════════════════════

if "pipeline_ran" not in st.session_state:
    st.session_state.pipeline_ran = False
if "baseline_profile" not in st.session_state:
    st.session_state.baseline_profile = None

if upload_base:
    with st.spinner("\u23f3 Training on base data..."):
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
            f.write(upload_base.getvalue())
            tmp = f.name
        base_df, base_fs = load_eeg_csv(tmp, fs_hint=int(fs_hint) if fs_hint > 0 else None)
        os.unlink(tmp)
        prof = fit_baseline(base_df, base_fs)
        st.session_state.baseline_profile = prof
        st.success(f"\u2705 Baseline Trained! (Max Amp: {prof.max_amplitude:.1f}\u00b5V, Max RMS: {prof.max_rms:.1f}\u00b5V)")

if run_btn or upload_test or not st.session_state.pipeline_ran:
    with st.spinner("\U0001f504 Running network & classification pipeline\u2026"):
        try:
            active_baseline = st.session_state.baseline_profile
            
            if upload_test:
                import tempfile, os
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
                    f.write(upload_test.getvalue())
                    tmp = f.name
                raw_df, detected_fs = load_eeg_csv(tmp, fs_hint=int(fs_hint) if fs_hint > 0 else None)
                os.unlink(tmp)
                real_fs    = int(fs_hint) if fs_hint > 0 else detected_fs
                n_channels = raw_df.channel.nunique()
                st.success(
                    f"\u2705 Loaded **{upload_test.name}** \u2014 "
                    f"{n_channels} channels \u00b7 "
                    f"fs \u2248 {real_fs} Hz"
                )
                result = run_pipeline_from_df(
                    raw_df.to_json(), real_fs,
                    loss, corruption, n_replay, n_fab, seed, baseline=active_baseline
                )
            else:
                result = run_pipeline(
                    duration, n_channels, loss, corruption, n_replay, n_fab, seed, fs=256, baseline=active_baseline
                )

            (df, windows, segs, ref, manifest, tx, verdicts,
             iscore, recs, cls_res, cls_df, recon, ev_df, active_fs) = result

            st.session_state.update(dict(
                df=df, windows=windows, segs=segs, ref=ref, manifest=manifest,
                tx=tx, verdicts=verdicts, iscore=iscore, recs=recs,
                cls_res=cls_res, cls_df=cls_df, recon=recon, ev_df=ev_df,
                active_fs=active_fs, pipeline_ran=True
            ))
        except Exception as e:
            import traceback
            st.error(f"Pipeline error: {e}")
            st.code(traceback.format_exc())
            st.stop()

S = st.session_state

# ═══════════════════════════════════════════════════════════════════════════════
# KPI Row
# ═══════════════════════════════════════════════════════════════════════════════

c1,c2,c3,c4,c5,c6 = st.columns(6)
cls_df  = S.cls_df
tx      = S.tx
iscore  = S.iscore

total    = len(cls_df)
bio      = (cls_df.verdict == CLASS_BIOLOGICAL).sum()
corr     = (cls_df.verdict == CLASS_CORRUPTION).sum()
tamp     = (cls_df.verdict == CLASS_TAMPER).sum()
clean    = (cls_df.verdict == CLASS_CLEAN).sum()

with c1: metric_card("Integrity Score",  f"{iscore:.0f}", "#6C63FF", integrity_badge(iscore))
with c2: metric_card("Total Segments",   total,           "#8892A4")
with c3: metric_card("Clean",            clean,           "#36D399")
with c4: metric_card("Biological",       bio,             "#00D4AA")
with c5: metric_card("Corrupted",        corr,            "#FFC947")
with c6: metric_card("Tampered",         tamp,            "#FF4E5B")

st.markdown("")

# ═══════════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 EEG Signals", "🔍 Event Timeline",
    "🌐 Transmission",  "🔐 Integrity Chain",
    "🧩 Classification","📋 Audit Log"
])

# ────────────────────────────────────────────────────────────────────
# TAB 1 — EEG Signals
# ────────────────────────────────────────────────────────────────────
with tab1:
    df      = S.df
    recon   = S.recon
    channels = df["channel"].unique().tolist()

    sel_ch = st.selectbox("Select channel", channels, key="sig_ch")

    orig_vals = df[df.channel == sel_ch]["value"].to_numpy()
    fs        = S.get("active_fs", 256)
    t_orig    = np.arange(len(orig_vals)) / fs

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("Original EEG","Reconstructed (Post-Transmission)","Residual / Error"),
                        vertical_spacing=0.08)

    recon_vals = recon.get(sel_ch, np.zeros_like(orig_vals))
    min_len    = min(len(orig_vals), len(recon_vals))
    t_short    = t_orig[:min_len]

    fig.add_trace(go.Scatter(x=t_short, y=orig_vals[:min_len],
                             line=dict(color="#6C63FF", width=1), name="Original"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_short, y=recon_vals[:min_len],
                             line=dict(color="#00D4AA", width=1), name="Reconstructed"), row=2, col=1)
    residual = orig_vals[:min_len] - recon_vals[:min_len]
    fig.add_trace(go.Scatter(x=t_short, y=residual,
                             line=dict(color="#FF4E5B", width=0.8), name="Residual",
                             fill="tozeroy", fillcolor="rgba(255,78,91,0.12)"), row=3, col=1)

    fig.update_layout(height=600, paper_bgcolor="#0F1117", plot_bgcolor="#13161f",
                      font=dict(color="#E2E8F0"), showlegend=True,
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(showgrid=True, gridcolor="#1e2235", title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="#1e2235")
    st.plotly_chart(fig, use_container_width=True)

    snr = snr_db(orig_vals[:min_len], recon_vals[:min_len])
    st.info(
        f"**SNR:** {snr:.1f} dB \u00a0|\u00a0 "
        f"**Samples:** {min_len} \u00a0|\u00a0 "
        f"**Duration:** {min_len/fs:.1f} s \u00a0|\u00a0 "
        f"**fs:** {fs} Hz \u00a0|\u00a0 "
        f"**Channel:** {sel_ch}"
    )

# ────────────────────────────────────────────────────────────────────
# TAB 2 — Event Timeline
# ────────────────────────────────────────────────────────────────────
with tab2:
    ev_df = S.ev_df
    hi    = ev_df[ev_df.priority == "HIGH"]

    fig2 = go.Figure()
    for ch in ev_df.channel.unique():
        sub = ev_df[ev_df.channel == ch]
        fig2.add_trace(go.Scatter(
            x=sub.t_start, y=[ch]*len(sub),
            mode="markers",
            marker=dict(
                size=sub.score / 8 + 4,
                color=sub.score,
                colorscale="Plasma",
                showscale=True,
                colorbar=dict(title="Priority Score", thickness=12),
                line=dict(width=0.5, color="#0F1117"),
            ),
            name=ch, hovertemplate=(
                "<b>%{y}</b><br>t=%{x:.2f}s<br>score=%{marker.color:.1f}<extra></extra>"
            )
        ))

    fig2.update_layout(
        title="Event Priority Timeline (bubble size ∝ score)",
        height=420, paper_bgcolor="#0F1117", plot_bgcolor="#13161f",
        font=dict(color="#E2E8F0"),
        xaxis=dict(title="Time (s)", showgrid=True, gridcolor="#1e2235"),
        yaxis=dict(title="Channel",  showgrid=False),
    )
    st.plotly_chart(fig2, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">HIGH Priority Windows</div>', unsafe_allow_html=True)
        st.dataframe(hi[["channel","t_start","t_end","score","events"]].head(30),
                     use_container_width=True, height=300)
    with col_b:
        st.markdown('<div class="section-header">Priority Score Distribution</div>', unsafe_allow_html=True)
        fig3 = px.histogram(ev_df, x="score", color="priority",
                            color_discrete_map={"HIGH":"#6C63FF","LOW":"#2D3748"},
                            nbins=30, barmode="overlay",
                            template="plotly_dark")
        fig3.update_layout(paper_bgcolor="#0F1117", plot_bgcolor="#13161f",
                           font=dict(color="#E2E8F0"), height=300, margin=dict(t=10))
        st.plotly_chart(fig3, use_container_width=True)

# ────────────────────────────────────────────────────────────────────
# TAB 3 — Transmission
# ────────────────────────────────────────────────────────────────────
with tab3:
    tx = S.tx
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: metric_card("Sent",       tx.sent_count,            "#8892A4")
    with col2: metric_card("Received",   len(tx.received),         "#36D399")
    with col3: metric_card("Dropped",    len(tx.dropped_ids),      "#FFC947")
    with col4: metric_card("Corrupted",  len(tx.corrupted_ids),    "#FF4E5B")
    with col5: metric_card("Replayed",   len(tx.replayed_ids),     "#FF4E5B")

    st.markdown("")

    labels  = ["Sent", "Received", "Dropped", "Corrupted", "Replayed", "Fabricated"]
    values  = [tx.sent_count, len(tx.received), len(tx.dropped_ids),
               len(tx.corrupted_ids), len(tx.replayed_ids), len(tx.fabricated_ids)]
    col_map = ["#6C63FF","#36D399","#FFC947","#FF4E5B","#FF4E5B","#FF1744"]

    fig4 = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=col_map,
        text=values, textposition="outside",
    ))
    fig4.update_layout(
        title="Network Event Breakdown",
        height=350, paper_bgcolor="#0F1117", plot_bgcolor="#13161f",
        font=dict(color="#E2E8F0"), showlegend=False,
        yaxis=dict(showgrid=True, gridcolor="#1e2235"),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Sim events table
    if tx.sim_events:
        rows = [{"segment_id": k, "events": ", ".join(v)} for k, v in tx.sim_events.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=250)

# ────────────────────────────────────────────────────────────────────
# TAB 4 — Integrity Chain
# ────────────────────────────────────────────────────────────────────
with tab4:
    verdicts = S.verdicts
    vdf = pd.DataFrame([{
        "segment_id":    v.segment_id,
        "channel":       v.channel,
        "t_start":       v.t_start,
        "hash_ok":       v.hash_ok,
        "chain_ok":      v.chain_ok,
        "verdict":       v.verdict,
        "stored_hash":   v.stored_hash,
        "computed_hash": v.computed_hash,
    } for v in verdicts])

    vc = vdf.verdict.value_counts().reset_index()
    vc.columns = ["verdict","count"]

    col_v = {"VALID":"#36D399","HASH_FAIL":"#FF4E5B","CHAIN_FAIL":"#FFC947","MISSING":"#8892A4"}
    figI = px.pie(vc, names="verdict", values="count",
                  color="verdict", color_discrete_map=col_v,
                  hole=0.55, template="plotly_dark")
    figI.update_layout(paper_bgcolor="#0F1117", font=dict(color="#E2E8F0"),
                       height=300, margin=dict(t=20,b=20))

    col_pie, col_chain = st.columns([1, 2])
    with col_pie:
        st.plotly_chart(figI, use_container_width=True)
        st.metric("Overall Integrity Score", f"{S.iscore:.1f} / 100",
                  delta=integrity_badge(S.iscore))
    with col_chain:
        st.markdown('<div class="section-header">Segment Hash Chain</div>', unsafe_allow_html=True)
        # Colour rows by verdict
        def colour_row(row):
            c = {"VALID":"rgba(54,211,153,0.15)","HASH_FAIL":"rgba(255,78,91,0.15)",
                 "CHAIN_FAIL":"rgba(255,201,71,0.15)"}.get(row.verdict,"")
            return [f"background:{c}"]*len(row)
        st.dataframe(vdf.style.apply(colour_row, axis=1), use_container_width=True, height=380)

# ────────────────────────────────────────────────────────────────────
# TAB 5 — Classification
# ────────────────────────────────────────────────────────────────────
with tab5:
    cls_df = S.cls_df

    # Verdict donut
    vc2 = cls_df.verdict.value_counts().reset_index()
    vc2.columns = ["verdict","count"]
    cmap2 = {CLASS_CLEAN:"#36D399", CLASS_BIOLOGICAL:"#00D4AA",
              CLASS_CORRUPTION:"#FFC947", CLASS_TAMPER:"#FF4E5B"}
    figC = px.pie(vc2, names="verdict", values="count",
                  color="verdict", color_discrete_map=cmap2,
                  hole=0.6, template="plotly_dark")
    figC.update_traces(textposition="outside", textinfo="percent+label")
    figC.update_layout(paper_bgcolor="#0F1117", font=dict(color="#E2E8F0"),
                       height=340, margin=dict(t=20,b=20))

    # Confidence scatter
    figS = px.scatter(cls_df, x="t_start", y="confidence",
                      color="verdict", color_discrete_map=cmap2,
                      size="priority_score", hover_data=["channel","rules"],
                      template="plotly_dark",
                      labels={"t_start":"Time (s)","confidence":"Confidence"})
    figS.update_layout(paper_bgcolor="#0F1117", plot_bgcolor="#13161f",
                       font=dict(color="#E2E8F0"), height=340,
                       legend=dict(bgcolor="rgba(0,0,0,0)"))

    col_d, col_s = st.columns(2)
    with col_d: st.plotly_chart(figC, use_container_width=True)
    with col_s: st.plotly_chart(figS, use_container_width=True)

    # Detailed table with coloured verdict column
    st.markdown('<div class="section-header">Segment-Level Verdicts</div>', unsafe_allow_html=True)
    display_df = cls_df[["segment_id","channel","t_start","t_end",
                          "verdict","confidence","priority","priority_score","rules"]].copy()

    def highlight_verdict(val):
        c = cmap2.get(val, "#fff")
        return f"color:{c};font-weight:600"

    styled = display_df.style.applymap(highlight_verdict, subset=["verdict"])
    st.dataframe(styled, use_container_width=True, height=380)

# ────────────────────────────────────────────────────────────────────
# TAB 6 — Audit Log
# ────────────────────────────────────────────────────────────────────
with tab6:
    recs = S.recs
    audit_rows = []
    for r in recs:
        audit_rows.append({
            "segment_id":    r.segment_id,
            "channel":       r.channel,
            "hash_ok":       "✅" if r.hash_ok  else "❌",
            "chain_ok":      "✅" if r.chain_ok else "❌",
            "amp_ok":        "✅" if r.amplitude_ok else "❌",
            "rms_ok":        "✅" if r.rms_ok  else "❌",
            "spec_ok":       "✅" if r.spectral_ok else "❌",
            "order_ok":      "✅" if r.order_ok else "❌",
            "replay":        "🔴" if r.is_replay else "—",
            "fabricated":    "🔴" if r.is_fabricated else "—",
            "rms":           r.rms_value,
            "peak2peak":     r.peak_to_peak,
            "dom_freq_hz":   r.dominant_freq,
            "evidence":      " | ".join(r.evidence) if r.evidence else "—",
        })
    adf = pd.DataFrame(audit_rows)

    search = st.text_input("🔍 Filter by segment ID or evidence", "")
    if search:
        mask = (adf.segment_id.str.contains(search, case=False) |
                adf.evidence.str.contains(search, case=False))
        adf = adf[mask]

    st.dataframe(adf, use_container_width=True, height=460)
    st.download_button("⬇ Download Audit CSV", adf.to_csv(index=False),
                       "trialflux_audit.csv", "text/csv")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<p style="text-align:center;color:#8892A4;font-size:0.8rem">
  TrialFlux · Neural Telemetry Integrity Engine · Biomedical Hackathon Prototype
  &nbsp;·&nbsp; Built with Streamlit, Plotly, SciPy &amp; Python
</p>
""", unsafe_allow_html=True)
