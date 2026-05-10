import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np
import community as community_louvain
import umap
from node2vec import Node2Vec
from helpers import load_graph, section

BG = '#1c1c1c'


def greedy_communities(G):
    return [set(c) for c in nx_comm.greedy_modularity_communities(G, weight='weight')]


def louvain_communities(G):
    partition = community_louvain.best_partition(G, weight='weight')
    groups = {}
    for node, cid in partition.items():
        groups.setdefault(cid, set()).add(node)
    return list(groups.values())


def label_propagation_communities(G):
    return [set(c) for c in nx_comm.label_propagation_communities(G)]


def node_color_map(communities):
    cmap = matplotlib.colormaps['hsv'].resampled(len(communities))
    return {node: cmap(i) for i, comm in enumerate(communities) for node in comm}


def compare_communities(G, greedy, louvain, label):
    names = ['Greedy Modularity', 'Louvain', 'Label Propagation']
    for name, comms in zip(names, [greedy, louvain, label]):
        sizes = [len(c) for c in comms]
        Q = nx_comm.modularity(G, comms, weight='weight')

        print(f"{name}")
        print(f"Communities:  {len(comms)}")
        print(f"Modularity Q: {Q:.4f}")
        print(f"Size min/max/mean/median: {min(sizes)} / {max(sizes)} / {np.mean(sizes):.1f} / {np.median(sizes):.1f} \n")


def plot_community_sizes(greedy, louvain, label):
    names = ['Greedy Modularity', 'Louvain', 'Label Propagation']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
    fig.suptitle('Community Size Distribution', fontsize=16, color='white')

    for ax, (name, comms) in zip(axes, zip(names, [greedy, louvain, label])):
        sizes = sorted([len(c) for c in comms], reverse=True)
        ax.bar(range(len(sizes)), sizes, color='steelblue', width=0.8)

        ax.set_facecolor(BG)
        ax.set_title(name, color='white', fontsize=12, pad=10)
        ax.set_xlabel('Community rank', color='white', fontsize=10)
        ax.set_ylabel('Nodes', color='white', fontsize=10)
        ax.tick_params(colors='white')
        ax.yaxis.grid(True, color='white', alpha=0.1, linestyle='--')
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')

        ax.text(0.97, 0.97, f'n={len(comms)}', transform=ax.transAxes,
                color='white', fontsize=9, ha='right', va='top', alpha=0.7)

    plt.tight_layout()
    path = os.path.join('data', 'community_sizes.png')
    plt.savefig(path, dpi=150, facecolor=BG, bbox_inches='tight')
    plt.close()


def draw_graph(G, img_path, communities, title='Witcher characters graph'):
    fig, ax = plt.subplots(figsize=(32, 32), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=36, color='white')
    ax.axis('off')

    node_color = node_color_map(communities)

    pos = nx.spring_layout(G, weight='weight', seed=42, k=18.0 / math.sqrt(G.number_of_nodes()))

    weighted_degree = dict(G.degree(weight='weight'))
    max_weighted_degree = max(weighted_degree.values())
    max_weight = max(d['weight'] for _, _, d in G.edges(data=True))

    edges = list(G.edges(data=True))
    edge_widths = [0.4 + 3.0 * d['weight'] / max_weight for _, _, d in edges]
    edge_alphas = [0.3 + 0.5 * d['weight'] / max_weight for _, _, d in edges]
    nx.draw_networkx_edges(G, pos, ax=ax,
                           width=edge_widths,
                           alpha=edge_alphas,
                           edge_color='white',
                           arrows=True,
                           arrowstyle='-',
                           connectionstyle='arc3,rad=0python.2')

    node_sizes = [50 + 2000 * (weighted_degree[n] / max_weighted_degree) ** 0.6 for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           alpha=0.95,
                           node_size=node_sizes,
                           node_color=[node_color[n] for n in G.nodes()])

    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color='white')

    ax.set_xlim(ax.get_xlim()[0] - 0.3, ax.get_xlim()[1] + 0.3)
    ax.set_ylim(ax.get_ylim()[0] - 0.3, ax.get_ylim()[1] + 0.3)
    plt.savefig(img_path, dpi=200, facecolor=BG, bbox_inches='tight')
    plt.close()
    print(f"Saved: {img_path}")


def draw_embedding(G, img_path, communities, title='Graph Embedding'):
    weighted_degree = dict(G.degree(weight='weight'))
    max_wd = max(weighted_degree.values())
    color_map = node_color_map(communities)

    n2v = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, seed=42, workers=1, quiet=True)
    model = n2v.fit()

    nodes = list(G.nodes())
    vectors = np.array([model.wv[str(n)] for n in nodes])

    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(vectors)

    colors = [color_map.get(n, (0.5, 0.5, 0.5, 1.0)) for n in nodes]
    sizes = [20 + 200 * (weighted_degree[n] / max_wd) ** 0.6 for n in nodes]

    fig, ax = plt.subplots(figsize=(32, 32), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=36, color='white')
    ax.axis('off')

    ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=sizes, alpha=0.8)

    top_nodes = sorted(nodes, key=lambda n: weighted_degree[n], reverse=True)[:10]
    node_idx = {n: i for i, n in enumerate(nodes)}
    for n in top_nodes:
        i = node_idx[n]
        ax.annotate(n, (embedding[i, 0], embedding[i, 1]),
                    color='white',
                    fontsize=8,
                    xytext=(4, 4),
                    textcoords='offset points')

    plt.savefig(img_path, dpi=200, facecolor=BG, bbox_inches='tight')
    plt.close()
    print(f"Saved: {img_path}")


if __name__ == '__main__':
    G = load_graph(os.path.join('data', 'connections.csv'))

    greedy = greedy_communities(G)
    louvain = louvain_communities(G)
    label = label_propagation_communities(G)

    section("Comparing communities")
    compare_communities(G, greedy, louvain, label)
    plot_community_sizes(greedy, louvain, label)

    # section("Greedy Modularity")
    # draw_graph(G, os.path.join('data', 'graph_greedy_drawen.png'), greedy, title='Greedy Modularity')
    # draw_embedding(G, os.path.join('data', 'graph_greedy_embedding.png'), greedy, title='Greedy Modularity')

    # section("Louvain")
    # draw_graph(G, os.path.join('data', 'graph_louvain.png'), louvain, title='Louvain')
    # draw_embedding(G, os.path.join('data', 'graph_louvain_embedding.png'), louvain, title='Louvain')

    # section("Label Propagation")
    # draw_graph(G, os.path.join('data', 'graph_label.png'), label, title='Label Propagation')
    # draw_embedding(G, os.path.join('data', 'graph_label_embedding.png'), label, title='Label Propagation')
