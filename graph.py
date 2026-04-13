import csv
import math
import os
import random
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


MIN_WEIGHT = 5
BG = '#1c1c1c'


def load_graph(csv_path):
    G = nx.Graph()
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            w = int(row['weight'])
            if w >= MIN_WEIGHT:
                G.add_edge(row['character1'], row['character2'], weight=w)
    return G


def draw_graph(G, img_path):
    fig, ax = plt.subplots(figsize=(32, 32), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_title('Witcher characters graph', fontsize=36, color='white')
    ax.axis('off')

    rng = random.Random(42)
    node_color = {n: (rng.random(), rng.random(), rng.random()) for n in G.nodes()}

    communities = list(nx_comm.greedy_modularity_communities(G, weight='weight'))
    cmap = matplotlib.colormaps['hsv'].resampled(len(communities))
    node_color = {node: cmap(i)
                for i, comm in enumerate(communities) for node in comm}

    pos = nx.spring_layout(G, weight='weight', seed=42, k=18.0 / math.sqrt(G.number_of_nodes()))

    weighted_degree = dict(G.degree(weight='weight'))
    max_weighted_degree = max(weighted_degree.values())
    max_weight = max(d['weight'] for _, _, d in G.edges(data=True))

    edges = list(G.edges(data=True))
    edge_widths = [0.4 + 3.0 * d['weight'] / max_weight for _, _, d in edges]
    edge_alphas = [0.3 + 0.5 * d['weight'] / max_weight for _, _, d in edges]
    nx.draw_networkx_edges(G, pos,
                            ax=ax,
                            width=edge_widths,
                            alpha=edge_alphas,
                            edge_color='white',
                            arrows=True,
                            connectionstyle='arc3,rad=0.2')

    node_sizes = [50 + 2000 * (weighted_degree[n] / max_weighted_degree) ** 0.6 for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos,
                            ax=ax,
                            alpha=0.95,
                            node_size=node_sizes,
                            node_color=[node_color[n] for n in G.nodes()],)

    nx.draw_networkx_labels(G, pos,
                            ax=ax,
                            font_size=7,
                            font_color='white')

    plt.savefig(img_path, dpi=200, facecolor=BG)
    plt.close()

if __name__ == '__main__':
    csv_path = os.path.join('data', 'connections.csv')
    img_path = os.path.join('data', 'graph.png')
    G = load_graph(csv_path)
    draw_graph(G, img_path)
