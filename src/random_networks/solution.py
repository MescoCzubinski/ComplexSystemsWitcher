import math
import os
import sys
import random
import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from utils import load_graph, section, DATA, BG, TOP_N, node_color_map, print_stats

os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)


def generate_er(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    p = (2 * m) / (n * (n - 1))
    R = nx.erdos_renyi_graph(n, p)
    for _, _, d in R.edges(data=True):
        d['weight'] = 1
    return R


def generate_ba(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    k = max(1, round(m / n))
    R = nx.barabasi_albert_graph(n, k)
    for _, _, d in R.edges(data=True):
        d['weight'] = 1
    return R


def describe_basic(G, name):
    section(f"{name} - basic properties")
    print(f"Nodes:                {G.number_of_nodes()}")
    print(f"Edges:                {G.number_of_edges()}")
    print(f"Density:              {nx.density(G):.6f}")
    print(f"Connected components: {nx.number_connected_components(G)}")

    largest = max(nx.connected_components(G), key=len)
    print(f"Largest component:    {len(largest)} nodes")

    G_cc = G.subgraph(largest)
    print(f"Average path length:  {nx.average_shortest_path_length(G_cc):.4f}")
    print(f"Avg clustering coeff: {nx.average_clustering(G):.4f}")


def describe_centralities(G, name):
    degree      = nx.degree_centrality(G)
    closeness   = nx.closeness_centrality(G)
    betweenness = nx.betweenness_centrality(G, normalized=True)
    edge_betw   = nx.edge_betweenness_centrality(G, normalized=True)

    node_sections = [
        ("degree",           degree),
        ("closeness",        closeness),
        ("node betweenness", betweenness),
    ]

    for title, data in node_sections:
        section(f"{name} - {title} distribution")
        values = list(data.values())
        print_stats(values)

    section(f"{name} - edge betweenness distribution")
    values = list(edge_betw.values())
    print_stats(values)

    return degree, closeness, betweenness, edge_betw


def describe_top_degrees(G, name):
    section(f"{name} - top nodes by degree")
    degree = dict(G.degree())
    for node, value in sorted(degree.items(), key=lambda x: x[1], reverse=True)[:TOP_N]:
        print(f"  {str(node):<20} {value}")


def describe_communities(G, name):
    comms = list(nx_comm.greedy_modularity_communities(G, weight='weight'))
    sizes = sorted([len(c) for c in comms], reverse=True)
    Q = nx_comm.modularity(G, comms, weight='weight')

    section(f"{name} - Greedy Modularity communities")
    print(f"Communities:              {len(comms)}")
    print(f"Modularity Q:             {Q:.4f}")
    print(f"Avg clustering coeff:     {nx.average_clustering(G):.4f}")
    print(f"Size min/max/mean/median: {min(sizes)} / {max(sizes)} / {np.mean(sizes):.1f} / {np.median(sizes):.1f}")
    print(f"Top sizes: {sizes[:10]}")

    return comms


def plot_degree_distribution(named_graphs, img_path):
    rows = []
    for name, G in named_graphs:
        counts = {}
        for _, d in G.degree():
            counts[d] = counts.get(d, 0) + 1
        for deg, cnt in counts.items():
            rows.append({'network': name, 'degree': deg, 'count': cnt})
    df = pd.DataFrame(rows)
    order = sorted(df['degree'].unique())

    fig, ax = plt.subplots(figsize=(16, 6), facecolor=BG)
    ax.set_facecolor(BG)

    sns.barplot(data=df, x='degree', y='count', hue='network',
                order=order, ax=ax, palette='Set2', alpha=0.85)

    title = ' vs '.join(n for n, _ in named_graphs)
    ax.set_title(f'degree distribution - {title}', color='white', fontsize=14, pad=10)
    ax.set_xlabel('degree', color='white', fontsize=10)
    ax.set_ylabel('count', color='white', fontsize=10)
    ax.tick_params(colors='white')
    ax.tick_params(axis='x', labelrotation=90, labelsize=8)
    ax.yaxis.grid(True, color='white', alpha=0.1, linestyle='--')
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_color('white')
        legend.get_frame().set_facecolor(BG)
        legend.get_frame().set_edgecolor('#444444')
        if legend.get_title() is not None:
            legend.get_title().set_color('white')

    plt.tight_layout()
    plt.savefig(img_path, dpi=150, facecolor=BG, bbox_inches='tight')
    plt.close()
    print(f"Saved: {img_path}")


def plot_community_sizes(named_comms, img_path):
    rows = []
    for name, comms in named_comms:
        sizes = sorted([len(c) for c in comms], reverse=True)
        for rank, size in enumerate(sizes):
            rows.append({'network': name, 'rank': rank, 'size': size})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    ax.set_facecolor(BG)

    sns.barplot(data=df, x='rank', y='size', hue='network',
                ax=ax, palette='Set2', alpha=0.85)

    ax.set_title('community sizes (Greedy Modularity)', color='white', fontsize=14, pad=10)
    ax.set_xlabel('community rank', color='white', fontsize=10)
    ax.set_ylabel('nodes', color='white', fontsize=10)
    ax.tick_params(colors='white')
    ax.yaxis.grid(True, color='white', alpha=0.1, linestyle='--')
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

    legend = ax.legend(facecolor=BG, edgecolor='#444444', labelcolor='white')
    legend.get_title().set_color('white')

    plt.tight_layout()
    plt.savefig(img_path, dpi=150, facecolor=BG, bbox_inches='tight')
    plt.close()
    print(f"Saved: {img_path}")


def draw_graph(G, img_path, communities, title, show_labels=False):
    fig, ax = plt.subplots(figsize=(32, 32), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=36, color='white')
    ax.axis('off')

    node_color = node_color_map(communities)

    pos = nx.spring_layout(G, weight='weight', seed=42, k=18.0 / math.sqrt(G.number_of_nodes()))

    weighted_degree = dict(G.degree(weight='weight'))
    max_weighted_degree = max(weighted_degree.values()) if weighted_degree else 1
    weights = [d.get('weight', 1) for _, _, d in G.edges(data=True)]
    if weights and min(weights) == max(weights):
        edge_widths, edge_alphas = 0.3, 0.25
    else:
        max_weight = max(weights)
        edge_widths = [0.4 + 3.0 * w / max_weight for w in weights]
        edge_alphas = [0.3 + 0.5 * w / max_weight for w in weights]
    nx.draw_networkx_edges(G, pos, ax=ax,
                           width=edge_widths,
                           alpha=edge_alphas,
                           edge_color='white',
                           arrows=True,
                           arrowstyle='-',
                           connectionstyle='arc3,rad=0.2')

    node_sizes = [50 + 2000 * (weighted_degree[n] / max_weighted_degree) ** 0.6 for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           alpha=0.95,
                           node_size=node_sizes,
                           node_color=[node_color[n] for n in G.nodes()])

    if show_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color='white')

    ax.set_xlim(ax.get_xlim()[0] - 0.3, ax.get_xlim()[1] + 0.3)
    ax.set_ylim(ax.get_ylim()[0] - 0.3, ax.get_ylim()[1] + 0.3)
    plt.savefig(img_path, dpi=200, facecolor=BG, bbox_inches='tight')
    plt.close()
    print(f"Saved: {img_path}")


if __name__ == '__main__':
    random.seed(37)
    np.random.seed(37)

    G    = load_graph(os.path.join(DATA, 'connections.csv'))
    G_er = generate_er(G)
    G_ba = generate_ba(G)

    describe_basic(G,    'Original')
    describe_basic(G_er, 'Erdős–Rényi')
    describe_basic(G_ba, 'Barabási-Albert')

    describe_top_degrees(G,    'Original')
    describe_top_degrees(G_er, 'Erdős–Rényi')
    describe_top_degrees(G_ba, 'Barabási-Albert')

    describe_centralities(G,    'Original')
    describe_centralities(G_er, 'Erdős–Rényi')
    describe_centralities(G_ba, 'Barabási-Albert')

    plot_degree_distribution([
        ('Original',        G),
        ('Erdős–Rényi',     G_er),
        ('Barabási-Albert', G_ba),
    ], os.path.join(HERE, 'figures/degree_distribution.png'))

    comms_original = describe_communities(G,    'Original')
    comms_er       = describe_communities(G_er, 'Erdős–Rényi')
    comms_ba       = describe_communities(G_ba, 'Barabási-Albert')

    plot_community_sizes([
        ('Original',        comms_original),
        ('Erdős–Rényi',     comms_er),
        ('Barabási-Albert', comms_ba),
    ], os.path.join(HERE, 'figures/community_sizes.png'))

    draw_graph(G,    os.path.join(HERE, 'figures/graph_original.png'), comms_original, 'Original network', show_labels=True)
    draw_graph(G_er, os.path.join(HERE, 'figures/graph_er.png'),       comms_er,       'Erdős–Rényi')
    draw_graph(G_ba, os.path.join(HERE, 'figures/graph_ba.png'),       comms_ba,       'Barabási-Albert')
