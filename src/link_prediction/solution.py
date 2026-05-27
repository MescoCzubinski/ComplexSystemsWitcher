import os
import sys
import glob
import random
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from utils import section, DATA, BG
DATA_DIR = os.path.join(DATA, 'email_data')


def load_graph(path):
    G = nx.Graph()
    with open(path) as f:
        for line in f:
            parts = line.strip().split(';')
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            G.add_edge(u, v)
    return G


def load_windows(prefix):
    pattern = os.path.join(DATA_DIR, f'{prefix}_*.txt')
    return sorted(glob.glob(pattern))


def split_to_historical_and_testset(paths):
    historical, testset = [], []
    for i, p in enumerate(paths):
        if (i + 1) % 5 == 0:
            testset.append(p)
        else:
            historical.append(p)
    return historical, testset


def build_cumulative_graph(paths):
    G = nx.Graph()
    for p in paths:
        G = nx.compose(G, load_graph(p))
    return G


def get_non_edges(G, sample_size=500):
    nodes = list(G.nodes())
    non_edges = []
    attempts = 0
    while len(non_edges) < sample_size and attempts < sample_size * 20:
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v):
            non_edges.append((u, v))
        attempts += 1
    return non_edges


def predict_common_neighbors(G, pairs):
    scores = []
    for u, v in pairs:
        if u not in G or v not in G:
            scores.append(0)
        else:
            scores.append(len(list(nx.common_neighbors(G, u, v))))
    return scores


def predict_jaccard(G, pairs):
    preds = {(u, v): p for u, v, p in nx.jaccard_coefficient(G, pairs)}
    return [preds.get(pair, 0.0) for pair in pairs]


def evaluate(scores, labels, threshold=None):
    scores = np.array(scores)
    labels = np.array(labels)

    if len(np.unique(labels)) < 2:
        return {'auc': None, 'precision': None, 'recall': None, 'f1': None}

    auc = roc_auc_score(labels, scores)
    if threshold is None:
        threshold = np.median(scores)

    preds = (scores >= threshold).astype(int)
    return {
        'auc':       auc,
        'precision': precision_score(labels, preds, zero_division=0),
        'recall':    recall_score(labels, preds, zero_division=0),
        'f1':        f1_score(labels, preds, zero_division=0),
    }


def describe_windows(prefix, name):
    paths = load_windows(prefix)
    historical, testset = split_to_historical_and_testset(paths)
    print(f"\n{name}")
    print(f"  Total:       {len(paths)}")
    print(f"  Historical:  {len(historical)}")
    print(f"  Test:        {len(testset)}")


def describe_results(name, results):
    valid = [r for r in results if r['auc'] is not None]
    print(f"\n  {name}")
    print(f"  {'Metric':<12} {'Mean':>10} {'Std':>10}")
    print("  " + "-" * 34)
    for metric in ['auc', 'precision', 'recall', 'f1']:
        vals = [r[metric] for r in valid if r[metric] is not None]
        print(f"  {metric:<12} {np.mean(vals):>10.4f} {np.std(vals):>10.4f}")


def describe_predictions(prefix, name):
    paths = load_windows(prefix)
    historical, testset = split_to_historical_and_testset(paths)
    print(f"\n{name}")
    print(f"  Total windows: {len(paths)}")
    print(f"  Historical:    {len(historical)}  Test: {len(testset)}")

    results_cn, results_jc = [], []
    historical_set = set(historical)
    hist_paths = []

    for p in paths:
        if p in historical_set:
            hist_paths.append(p)
            continue
        if not hist_paths or len(results_cn) >= 10:
            continue

        G_hist = build_cumulative_graph(hist_paths)
        G_test = load_graph(p)

        positive_pairs = [(u, v) for u, v in G_test.edges()
                          if not G_hist.has_edge(u, v)
                          and u in G_hist and v in G_hist][:100]

        if len(positive_pairs) < 5:
            continue

        negative_pairs = get_non_edges(G_hist, sample_size=len(positive_pairs))
        pairs = positive_pairs + negative_pairs
        labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

        cn_scores = predict_common_neighbors(G_hist, pairs)
        jc_scores = predict_jaccard(G_hist, pairs)

        results_cn.append(evaluate(cn_scores, labels))
        results_jc.append(evaluate(jc_scores, labels))

    describe_results("Common Neighbors", results_cn)
    describe_results("Jaccard",          results_jc)


def plot_edge_counts(prefix, name):
    paths = load_windows(prefix)
    counts = np.array([load_graph(p).number_of_edges() for p in paths])

    fig, ax = plt.subplots(figsize=(16, 5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.plot(counts, color='steelblue', linewidth=0.8)

    ax.set_title(f'Edges per window - {name}', color='white', fontsize=14)
    ax.set_xlabel('Window index', color='white')
    ax.set_ylabel('Edges', color='white')
    ax.tick_params(colors='white')
    ax.yaxis.grid(True, color='white', alpha=0.1, linestyle='--')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

    path = os.path.join(HERE, f'edge_counts_{prefix}.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=BG, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_predictions(prefix, name, predictor, predictor_name,
                     top_n=200, top_k=50):
    paths = load_windows(prefix)
    historical, testset = split_to_historical_and_testset(paths)
    if not testset:
        return

    test_path = testset[len(testset) // 2]
    idx = paths.index(test_path)
    hist_paths = [p for p in historical if paths.index(p) < idx]
    if not hist_paths:
        return

    G_hist = build_cumulative_graph(hist_paths)
    G_test = load_graph(test_path)

    top_nodes = sorted(G_hist.nodes(),
                       key=lambda n: G_hist.degree(n), reverse=True)[:top_n]
    top_set = set(top_nodes)
    S_hist = G_hist.subgraph(top_nodes).copy()

    candidates = [(u, v) for i, u in enumerate(top_nodes)
                  for v in top_nodes[i + 1:] if not G_hist.has_edge(u, v)]
    scores = predictor(G_hist, candidates)
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    predicted = {frozenset(p) for p, _ in ranked[:top_k]}

    actual_new = {frozenset((u, v)) for u, v in G_test.edges()
                  if not G_hist.has_edge(u, v)
                  and u in top_set and v in top_set}

    hits         = predicted & actual_new
    pred_only    = predicted - actual_new
    actual_only  = actual_new - predicted

    print(f"  [{predictor_name} / {prefix}] "
          f"candidates={len(candidates)}  "
          f"predicted={len(predicted)}  "
          f"actual_new={len(actual_new)}  "
          f"hits={len(hits)}")

    pos = nx.spring_layout(S_hist, seed=42, k=2.0)

    fig, ax = plt.subplots(figsize=(14, 10), facecolor=BG)
    ax.set_facecolor(BG)
    title = (f'{predictor_name} - {name}\n'
             f'green=predicted, red=missed, gold=hit ({len(hits)}/{top_k})')
    ax.set_title(title, color='white', fontsize=12)
    ax.axis('off')

    nx.draw_networkx_edges(S_hist, pos, ax=ax,
                           edge_color='white', alpha=0.15, width=0.4)

    sizes = [S_hist.degree(n) * 6 + 40 for n in S_hist.nodes()]
    nx.draw_networkx_nodes(S_hist, pos, ax=ax, node_size=sizes,
                           node_color='steelblue', alpha=0.85)

    def _draw(edges, color, width):
        if edges:
            nx.draw_networkx_edges(S_hist, pos, ax=ax,
                                   edgelist=[tuple(e) for e in edges],
                                   edge_color=color, width=width, alpha=0.9)
    _draw(pred_only,   'limegreen', 2.0)
    _draw(actual_only, 'red',       2.0)
    _draw(hits,        'gold',      3.0)

    out = os.path.join(HERE,
                       f'predictions_{prefix}_{predictor_name.lower()}.png')
    plt.savefig(out, dpi=150, facecolor=BG, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    random.seed(42)

    section("Window splits")
    describe_windows('o1p1',   'Daily windows')
    describe_windows('o30p15', '30-day windows')

    section("Predictions")
    describe_predictions('o1p1', 'Daily (o1p1)')
    describe_predictions('o30p15', '30-day (o30p15)')

    section("Edge counts over time")
    plot_edge_counts('o1p1',   'Daily (o1p1)')
    plot_edge_counts('o30p15', '30-day (o30p15)')

    section("Prediction visualization")
    plot_predictions('o1p1',   'Daily (o1p1)',   predict_common_neighbors, 'CN')
    plot_predictions('o1p1',   'Daily (o1p1)',   predict_jaccard,          'Jaccard')
    plot_predictions('o30p15', '30-day (o30p15)', predict_common_neighbors, 'CN')
    plot_predictions('o30p15', '30-day (o30p15)', predict_jaccard,          'Jaccard')

