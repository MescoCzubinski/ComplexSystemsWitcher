import csv
import math
import os
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

CONNECTIONS_DIR = 'data'
MIN_WEIGHT = 5

PALETTE = [
    '#4e9af1', '#f5a623', '#7ed321', '#e74c3c',
    '#9b59b6', '#f1c40f', '#e91e8c', '#1abc9c',
    '#e67e22', '#00bcd4',
]
BG = '#1c1c1e'


def load_graph(csv_path):
    G = nx.Graph()
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            w = int(row['weight'])
            if w >= MIN_WEIGHT:
                G.add_edge(row['character1'], row['character2'], weight=w)
    return G


def _curved_edge(ax, x1, y1, x2, y2, color, lw, alpha, rad=0.2):
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1e-9
    cx = mx - dy / length * rad * length * 0.5
    cy = my + dx / length * rad * length * 0.5
    path = Path([(x1, y1), (cx, cy), (x2, y2)],
                [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    ax.add_patch(PathPatch(path, facecolor='none', edgecolor=color,
                           linewidth=lw, alpha=alpha))


def draw_graph(G, title, out_path):
    fig, ax = plt.subplots(figsize=(32, 32), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=16, color='white', pad=20)
    ax.axis('off')

    communities = list(nx_comm.greedy_modularity_communities(G, weight='weight'))
    node_community = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_community[node] = i
    node_color = {n: PALETTE[node_community[n] % len(PALETTE)] for n in G.nodes()}

    pos = nx.spring_layout(G, weight='weight', seed=42,
                           k=10.0 / math.sqrt(G.number_of_nodes()))

    weighted_degree = dict(G.degree(weight='weight'))
    max_wd = max(weighted_degree.values())
    max_w = max(d['weight'] for _, _, d in G.edges(data=True))

    for u, v, d in G.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        color = node_color[u] if weighted_degree[u] >= weighted_degree[v] else node_color[v]
        lw = 0.4 + 3.0 * d['weight'] / max_w
        alpha = 0.25 + 0.55 * d['weight'] / max_w
        _curved_edge(ax, x1, y1, x2, y2, color, lw, alpha)

    node_sizes = [80 + 3000 * (weighted_degree[n] / max_wd) ** 0.65 for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_size=node_sizes,
                           node_color=[node_color[n] for n in G.nodes()],
                           alpha=0.95)

    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6, font_color='white')

    ax.set_xlim(ax.get_xlim()[0] - 0.1, ax.get_xlim()[1] + 0.1)
    ax.set_ylim(ax.get_ylim()[0] - 0.1, ax.get_ylim()[1] + 0.1)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  saved {out_path}  ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)')


if __name__ == '__main__':
    csv_path = os.path.join(CONNECTIONS_DIR, 'connections.csv')
    out_path = os.path.join(CONNECTIONS_DIR, 'graph.png')
    G = load_graph(csv_path)
    draw_graph(G, 'Witcher character co-occurrence  (window = 2 sentences)', out_path)
