# Ricci Flow Engine

Applies discrete Ricci flow to the ETF correlation graph, computing multiple curvature measures (Ollivier‑Ricci, Forman‑Ricci, entropy curvature, geodesic deviation). The graph is evolved by updating edge weights based on curvature, then each ETF's geometric instability score is the sum of absolute incident curvatures. Global signals include curvature stress index, momentum, entropy, and geodesic deviation.

- **Curvatures:** Ollivier‑Ricci, Forman‑Ricci, entropy, geodesic deviation
- **Flow:** Discrete Ricci flow (5 iterations, step=0.1)
- **Windows:** 63, 252, 504, 1008, 2016 days (best per ETF)
- **Output:** top 3 ETFs per universe by instability score, global metrics

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
