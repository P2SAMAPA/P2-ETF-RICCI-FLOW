import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import networkx as nx
import config
import data_manager
from ricci_flow import (forman_ricci_curvature, ollivier_ricci_curvature,
                        entropy_curvature, geodesic_deviation, ricci_flow)

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    return obj

def compute_etf_instability(G, curv):
    """
    Compute instability as sum of |curvature| * weight / (degree+1) for each node.
    """
    instability = {}
    for node in G.nodes:
        total = 0.0
        deg = G.degree(node, weight='weight')
        for nb in G.neighbors(node):
            edge = (node, nb) if (node, nb) in curv else (nb, node)
            c = curv.get(edge, 0.0)
            w = G[node][nb]['weight']
            total += abs(c) * w
        instability[node] = total / (deg + 1.0)
    return instability

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Ricci Flow) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < max(config.WINDOWS) + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        best_per_etf = {}
        window_results = {}

        for win in config.WINDOWS:
            if len(returns) < win + 2:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            print(f"  Processing window {win}d...")
            ret_win = returns.iloc[-win:]
            # Build correlation graph with sparsification (keep only top 20% edges by correlation)
            corr = ret_win.corr().abs()
            G = nx.Graph()
            for ticker in tickers:
                G.add_node(ticker)
            # Keep edges where correlation > 0.5 to sparsify
            threshold = 0.5
            for i in range(len(tickers)):
                for j in range(i+1, len(tickers)):
                    w = corr.iloc[i, j]
                    if w > threshold:
                        G.add_edge(tickers[i], tickers[j], weight=w)
            # If graph is empty, use all edges
            if G.number_of_edges() == 0:
                for i in range(len(tickers)):
                    for j in range(i+1, len(tickers)):
                        w = corr.iloc[i, j]
                        if w > 0:
                            G.add_edge(tickers[i], tickers[j], weight=w)
            # Compute curvatures on original graph
            ollivier = ollivier_ricci_curvature(G)
            forman = forman_ricci_curvature(G)
            entropy = entropy_curvature(G)
            geodesic = geodesic_deviation(G)
            # Apply Ricci flow
            G_flow = ricci_flow(G, iterations=config.RICCI_ITERATIONS, step=config.RICCI_STEP)
            # Compute curvatures after flow (use Ollivier again)
            curv_flow = ollivier_ricci_curvature(G_flow)
            # Compute instability scores
            instability = compute_etf_instability(G_flow, curv_flow)
            # Stress index = average absolute curvature after flow
            stress_index = np.mean([abs(c) for c in curv_flow.values()]) if curv_flow else 0.0
            # Curvature momentum = change in average curvature (placeholder)
            momentum = 0.0
            window_results[win] = {
                "instability": instability,
                "stress_index": stress_index,
                "momentum": momentum,
                "entropy": entropy,
                "geodesic_deviation": geodesic
            }
            for etf, score in instability.items():
                if etf not in best_per_etf or score > best_per_etf[etf][0]:
                    best_per_etf[etf] = (score, win)

        if not best_per_etf:
            print("  No valid predictions – falling back to historical mean return")
            for etf in tickers:
                if etf in returns.columns:
                    mean_ret = returns[etf].iloc[-252:].mean()
                    if not np.isnan(mean_ret):
                        best_per_etf[etf] = (max(mean_ret, 1e-6), 0)
            if not best_per_etf:
                all_results[universe_name] = {"top_etfs": []}
                continue

        full_scores = {ticker: {"score": float(score), "best_window": win} for ticker, (score, win) in best_per_etf.items()}
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = [{"ticker": ticker, "instability": float(score), "best_window": win} for ticker, (score, win) in sorted_etfs[:config.TOP_N]]

        print(f"  Top 3 ETFs by geometric instability: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "window_results": window_results,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/ricci_flow_{today}.json")
    with open(local_path, "w") as f:
        json.dump(convert_to_serializable({"run_date": today, "universes": all_results}), f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Ricci Flow Engine complete ===")

if __name__ == "__main__":
    main()
