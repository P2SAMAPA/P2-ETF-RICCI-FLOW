import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Ricci Flow Engine", layout="wide")
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #555; margin-bottom: 2rem; }
    .universe-title { font-size: 1.5rem; font-weight: 600; margin-top: 1rem; margin-bottom: 1rem; padding-left: 0.5rem; border-left: 5px solid #1f77b4; }
    .etf-card { background: linear-gradient(135deg, #1f77b4 0%, #2c3e50 100%); color: white; border-radius: 15px; padding: 1rem; margin: 0.5rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
    .etf-ticker { font-size: 1.3rem; font-weight: bold; }
    .etf-score { font-size: 0.9rem; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌀 Ricci Flow Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ollivier‑Ricci curvature | Forman‑Ricci | Entropy curvature | Geodesic deviation | Discrete Ricci flow | Multi‑window evaluation</div>', unsafe_allow_html=True)

st.sidebar.markdown("## 🌀 Ricci Flow")
st.sidebar.markdown(f"**Run Date:** `{st.session_state.get('run_date', 'Not loaded')}`")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Ricci iterations:** {config.RICCI_ITERATIONS} | **Step:** {config.RICCI_STEP}")
st.sidebar.markdown("**Windows evaluated:** 63, 252, 504, 1008, 2016 days (best per ETF)")

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'ricci_flow_' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error: {data['error']}")
    st.stop()

st.session_state['run_date'] = data['run_date']
universes = data["universes"]

st.header("🏆 Top ETFs by Geometric Instability Score (after Ricci flow)")

# Explanation section
with st.expander("📖 How to interpret the metrics", expanded=True):
    st.markdown("""
    - **Geometric Instability Score (per ETF):** Sum of absolute incident curvatures weighted by edge strength. **Higher score** indicates the ETF sits in a geometrically stressed region of the market – often a precursor to volatility spikes or regime shifts. **Ranking:** pick ETFs with **highest** instability.
    - **Stress Index (global):** Average absolute curvature across all edges. High stress = the whole market is tense.
    - **Curvature Momentum:** Change in stress index (positive = increasing stress, negative = relaxing). Currently a placeholder; future versions will track historical changes.
    - **Entropy Curvature:** Von Neumann entropy of the graph Laplacian. Higher entropy = more disordered/less curved market.
    - **Geodesic Deviation:** Variance of shortest path lengths. Higher deviation = more heterogeneous distances → more curved geometry.
    
    **How to pick the best ETF:** Sort ETFs by **Instability Score** (highest first). The top ETFs are those most geometrically stressed, which often lead market moves or are early indicators of systemic risk.
    """)

for universe_name, uni_data in universes.items():
    top_etfs = uni_data.get("top_etfs", [])
    if not top_etfs:
        continue
    st.markdown(f'<div class="universe-title">{universe_name.replace("_", " ").title()}</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, etf in enumerate(top_etfs):
        with cols[idx]:
            st.markdown(f"""
            <div class="etf-card">
                <div class="etf-ticker">{etf['ticker']}</div>
                <div class="etf-score">instability = {etf['instability']:.4f}</div>
                <div class="etf-score">best window = {etf.get('best_window', 'N/A')}d</div>
            </div>
            """, unsafe_allow_html=True)
    # Global metrics for the best window
    global_metrics = uni_data.get("global_metrics", {})
    if global_metrics:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Stress Index", f"{global_metrics.get('stress_index', 0):.4f}")
        with col2:
            st.metric("Curvature Momentum", f"{global_metrics.get('momentum', 0):.4f}")
        with col3:
            st.metric("Entropy Curvature", f"{global_metrics.get('entropy', 0):.4f}")
        with col4:
            st.metric("Geodesic Deviation", f"{global_metrics.get('geodesic_deviation', 0):.4f}")
    else:
        st.info("Global metrics not available for this window.")
    with st.expander("📋 Full ranking (all ETFs, best window per ETF)"):
        full = uni_data.get("full_scores", {})
        if full:
            rows = []
            for ticker, info in full.items():
                if isinstance(info, dict):
                    score = info.get("score", 0.0)
                    win = info.get("best_window", "N/A")
                else:
                    score = info
                    win = "N/A"
                rows.append({"ETF": ticker, "Instability Score": score, "Best Window": win})
            df = pd.DataFrame(rows)
            df["Instability Score"] = pd.to_numeric(df["Instability Score"], errors='coerce')
            df = df.dropna(subset=["Instability Score"]).sort_values("Instability Score", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
    st.divider()

st.caption("The correlation graph is evolved via discrete Ricci flow (5 iterations). Instability score = sum of |curvature| × weight / (degree+1). Higher instability = more geometrically stressed ETF. Global metrics are computed on the full correlation graph for the window where the top ETF performed best.")
