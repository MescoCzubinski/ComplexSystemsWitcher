import math
import os
import random
import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from helpers import load_graph, section


BG = '#1c1c1c'
TOP_N = 10


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
        ("Degree",           degree),
        ("Closeness",        closeness),
        ("Node Betweenness", betweenness),
    ]

    for title, data in node_sections:
        section(f"{name} - {title} distribution")
        values = list(data.values())
        print(f"  Min:    {min(values):.4f}")
        print(f"  Max:    {max(values):.4f}")
        print(f"  Mean:   {np.mean(values):.4f}")
        print(f"  Median: {np.median(values):.4f}")
        print(f"  Std:    {np.std(values):.4f}")

    section(f"{name} - Edge Betweenness distribution")
    values = list(edge_betw.values())
    print(f"  Min:    {min(values):.4f}")
    print(f"  Max:    {max(values):.4f}")
    print(f"  Mean:   {np.mean(values):.4f}")
    print(f"  Median: {np.median(values):.4f}")
    print(f"  Std:    {np.std(values):.4f}")

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


def plot_degree_distribution(G, name, img_path):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)

    degrees = [d for _, d in G.degree()]
    bins = range(0, max(degrees) + 2)
    ax.hist(degrees, bins=bins, color='steelblue', edgecolor='#1c1c1c', alpha=0.9)

    ax.set_facecolor(BG)
    ax.set_title(f'Degree distribution - {name}', color='white', fontsize=14, pad=10)
    ax.set_xlabel('Degree', color='white', fontsize=10)
    ax.set_ylabel('Count', color='white', fontsize=10)
    ax.tick_params(colors='white')
    ax.yaxis.grid(True, color='white', alpha=0.1, linestyle='--')
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

    plt.tight_layout()
    plt.savefig(img_path, dpi=150, facecolor=BG, bbox_inches='tight')
    plt.close()
    print(f"Saved: {img_path}")


def plot_community_sizes(named_comms, img_path):
    rows = []
    for name, comms in named_comms:
        sizes = sorted([len(c) for c in comms], reverse=True)
        for rank, size in enumerate(sizes):
            rows.append({'Network': name, 'Rank': rank, 'Size': size})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    ax.set_facecolor(BG)

    sns.barplot(data=df, x='Rank', y='Size', hue='Network',
                ax=ax, palette='Set2', alpha=0.85)

    ax.set_title('Community sizes (Greedy Modularity)', color='white', fontsize=14, pad=10)
    ax.set_xlabel('Community rank', color='white', fontsize=10)
    ax.set_ylabel('Nodes', color='white', fontsize=10)
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


def node_color_map(communities):
    cmap = matplotlib.colormaps['hsv'].resampled(len(communities))
    return {node: cmap(i) for i, comm in enumerate(communities) for node in comm}


def draw_graph(G, img_path, communities, title, show_labels=False):
    fig, ax = plt.subplots(figsize=(32, 32), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=36, color='white')
    ax.axis('off')

    node_color = node_color_map(communities)

    pos = nx.spring_layout(G, weight='weight', k=18.0 / math.sqrt(G.number_of_nodes()))

    degree = dict(G.degree())
    max_degree = max(degree.values()) if degree else 1
    max_weight = max((d.get('weight', 1) for _, _, d in G.edges(data=True)), default=1)

    edge_widths = [0.4 + 3.0 * d.get('weight', 1) / max_weight for _, _, d in G.edges(data=True)]
    edge_alphas = [0.3 + 0.5 * d.get('weight', 1) / max_weight for _, _, d in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, ax=ax,
                           width=edge_widths,
                           alpha=edge_alphas,
                           edge_color='white')

    node_sizes = [50 + 2000 * (degree[n] / max_degree) ** 0.6 for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           alpha=0.95,
                           node_size=node_sizes,
                           node_color=[node_color[n] for n in G.nodes()])

    if show_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color='white')

    plt.savefig(img_path, dpi=200, facecolor=BG, bbox_inches='tight')
    plt.close()
    print(f"Saved: {img_path}")


if __name__ == '__main__':
    random.seed(37)
    np.random.seed(37)

    G    = load_graph(os.path.join('data', 'connections.csv'))
    G_er = generate_er(G)
    G_ba = generate_ba(G)

    describe_basic(G,    'Orginal')
    describe_basic(G_er, 'Erdős–Rényi')
    describe_basic(G_ba, 'Barabási-Albert')

    describe_top_degrees(G,    'Orginal')
    describe_top_degrees(G_er, 'Erdős–Rényi')
    describe_top_degrees(G_ba, 'Barabási-Albert')

    describe_centralities(G,    'Orginal')
    describe_centralities(G_er, 'Erdős–Rényi')
    describe_centralities(G_ba, 'Barabási-Albert')

    plot_degree_distribution(G,    'Orginal',         os.path.join('data', 'ex5_degree_orginal.png'))
    plot_degree_distribution(G_er, 'Erdős–Rényi',     os.path.join('data', 'ex5_degree_er.png'))
    plot_degree_distribution(G_ba, 'Barabási-Albert', os.path.join('data', 'ex5_degree_ba.png'))

    comms_orginal = describe_communities(G,    'Orginal')
    comms_er      = describe_communities(G_er, 'Erdős–Rényi')
    comms_ba      = describe_communities(G_ba, 'Barabási-Albert')

    plot_community_sizes([
        ('Orginal',         comms_orginal),
        ('Erdős–Rényi',     comms_er),
        ('Barabási-Albert', comms_ba),
    ], os.path.join('data', 'ex5_community_sizes.png'))

    draw_graph(G,    os.path.join('data', 'ex5_graph_orginal.png'), comms_orginal, 'Orginal network', show_labels=True)
    draw_graph(G_er, os.path.join('data', 'ex5_graph_er.png'),      comms_er,      'Erdős–Rényi')
    draw_graph(G_ba, os.path.join('data', 'ex5_graph_ba.png'),      comms_ba,      'Barabási-Albert')
