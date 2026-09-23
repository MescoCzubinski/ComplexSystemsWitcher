import math
import os
import sys
import random
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from utils import load_graph, section, DATA, TOP_N

os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)

FG = 'black'
PANEL = 'white'

sns.set_theme(style='white')
PALETTE = sns.color_palette()

TOP_FRAC = 0.01


def generate_ba(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    k = max(1, round(m / n))
    R = nx.barabasi_albert_graph(n, k)
    for _, _, d in R.edges(data=True):
        d['weight'] = 1
    return R


def compute_hubness(G):
    deg = dict(G.degree())
    order = sorted(deg, key=lambda n: deg[n])
    n = len(order)
    if n <= 1:
        return {node: 1.0 for node in order}
    return {node: rank / (n - 1) for rank, node in enumerate(order)}


def detour_hops(G, u, v):
    data = dict(G[u][v])
    G.remove_edge(u, v)
    try:
        d = nx.shortest_path_length(G, u, v)
    except nx.NetworkXNoPath:
        d = None
    G.add_edge(u, v, **data)
    return d


def detect_anomalies(G, top_frac=TOP_FRAC):
    hub = compute_hubness(G)
    rows = []
    for u, v in list(G.edges()):
        d = detour_hops(G, u, v)
        if d is None:
            continue
        periph = (1.0 - hub[u]) * (1.0 - hub[v])
        rows.append({'u': u, 'v': v, 'hub_u': hub[u], 'hub_v': hub[v], 'hops': d, 'periph': periph, 'score': d * periph})

    rows.sort(key=lambda r: r['score'], reverse=True)
    n_anom = round(top_frac * len(rows))
    thr = rows[n_anom - 1]['score'] if 0 < n_anom <= len(rows) else float('inf')
    for i, r in enumerate(rows):
        r['significant'] = i < n_anom

    return {'hubness': hub, 'edges': rows, 'top_frac': top_frac, 'threshold': thr}


def describe_hubness(hub, name):
    section(f"{name} - top hubs (by hubness)")
    wd = sorted(hub.items(), key=lambda x: x[1], reverse=True)
    for node, val in wd[:TOP_N]:
        print(f"  {str(node):<26} {val:.4f}")


def describe_anomalies(result, name):
    rows = result['edges']
    sig = [r for r in rows if r['significant']]

    section(f"{name} - anomaly summary")
    print(f"Scored edges:           {len(rows)}")
    print(f"Top fraction:           {result['top_frac']:.0%}")
    print(f"Score threshold:        {result['threshold']:.3f}")
    print(f"Anomalies:              {len(sig)}")

    section(f"{name} - top anomalous edges")
    header = (f"{'character 1':<24}{'character 2':<24}"
              f"{'hub_u':>8}{'hub_v':>8}{'hops':>6}{'periph':>9}{'score':>9}")
    print(header)
    for r in rows[:TOP_N]:
        print(f"{str(r['u']):<24}{str(r['v']):<24}{r['hub_u']:>8.3f}{r['hub_v']:>8.3f}{int(r['hops']):>6}{r['periph']:>9.3f}{r['score']:>9.3f}")


def plot_anomaly_scores(result, img_path, title):
    scores = [r['score'] for r in result['edges']]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=range(len(scores)), y=scores, ax=ax, color=PALETTE[0])
    ax.axhline(result['threshold'], color=PALETTE[3], linestyle='--', label=f'threshold (top {result["top_frac"]:.0%})')
    ax.set(title=title, xlabel='edge (by score rank)', ylabel='anomaly score')
    ax.legend()
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {img_path}")


def draw_anomalies(G, result, img_path, title):
    sig = {tuple(sorted((r['u'], r['v']))) for r in result['edges'] if r['significant']}

    fig, ax = plt.subplots(figsize=(32, 32), facecolor=PANEL)
    ax.set_facecolor(PANEL)
    ax.set_title(title, fontsize=36, color=FG)
    ax.axis('off')

    pos = nx.spring_layout(G, weight='weight', seed=42, k=18.0 / math.sqrt(G.number_of_nodes()))

    normal = [(u, v) for u, v in G.edges() if tuple(sorted((u, v))) not in sig]
    anom = [(u, v) for u, v in G.edges() if tuple(sorted((u, v))) in sig]

    nx.draw_networkx_edges(G, pos, ax=ax,
                            edgelist=normal,
                            width=0.4,
                            alpha=0.15,
                            edge_color=PALETTE[7])

    nx.draw_networkx_edges(G, pos, ax=ax,
                            edgelist=anom,
                            width=2.5,
                            alpha=0.9,
                            edge_color=PALETTE[3])

    nx.draw_networkx_nodes(G, pos, ax=ax,
                            node_size=60,
                            node_color=matplotlib.colors.to_hex(PALETTE[0]))

    ax.set_xlim(ax.get_xlim()[0] - 0.3, ax.get_xlim()[1] + 0.3)
    ax.set_ylim(ax.get_ylim()[0] - 0.3, ax.get_ylim()[1] + 0.3)
    plt.savefig(img_path, dpi=200, facecolor=PANEL, bbox_inches='tight')
    plt.close()
    print(f"Saved: {img_path}")


if __name__ == '__main__':
    random.seed(37)
    np.random.seed(37)

    G = load_graph(os.path.join(DATA, 'connections.csv'))
    G_ba = generate_ba(G)

    for graph, name, suffix in [(G, 'Original', 'original'), (G_ba, 'Barabási-Albert', 'ba')]:
        result = detect_anomalies(graph)

        describe_hubness(result['hubness'], name)
        describe_anomalies(result, name)

        plot_anomaly_scores(result, os.path.join(HERE, f'figures/anomaly_scores_{suffix}.png'), f'{name} - anomaly scores')
        draw_anomalies(graph, result, os.path.join(HERE, f'figures/graph_anomalies_{suffix}.png'), f'{name} - anomalous edges')
