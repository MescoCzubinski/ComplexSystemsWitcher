import csv
import os
import sys
import math
import random
import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from utils import load_graph, section, DATA, BG, TOP_N


MIN_WEIGHT = 5
MATRIX_SUBGRAPH_SIZE = 15


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


def describe_graph_type(G):
    results = []

    if G.is_directed():
        results.append("directed")
    else:
        results.append("undirected")

    if G.is_multigraph():
        results.append("multiple edges allowed")
    else:
        results.append("simple")

    if nx.is_weighted(G):
        results.append("weighted")
    else:
        results.append("unweighted")

    if G.is_directed():
        connected = nx.is_weakly_connected(G)
    else:
        connected = nx.is_connected(G)
    results.append("connected" if connected else "disconnected")

    print("The graph is:")
    for r in results:
        print(" -", r)


def describe_graph_size(G):
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges:  {G.number_of_edges()}")


def describe_node_centrality(G):
    degree      = nx.degree_centrality(G)
    closeness   = nx.closeness_centrality(G, distance='weight')
    betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)

    sections = [
        ("Degree",      degree),
        ("Closeness",   closeness),
        ("Betweenness", betweenness),
    ]

    for title, data in sections:
        section(title)
        for node, value in data.items():
            print(f"{node:<40} {value:.4f}")

    return degree, closeness, betweenness


def describe_edge_centrality(G):
    betweenness = nx.edge_betweenness_centrality(G, weight='weight', normalized=True)

    print("\n=== Edge Betweenness ===")
    for (u, v), value in betweenness.items():
        print(f"{u:<20} - {v:<20} {value:.4f}")

    return betweenness


def describe_top(G, top_n=10):
    degree          = nx.degree_centrality(G)
    closeness       = nx.closeness_centrality(G, distance='weight')
    betweenness     = nx.betweenness_centrality(G, weight='weight', normalized=True)
    edge_betweenness = nx.edge_betweenness_centrality(G, weight='weight', normalized=True)

    sections = [
        ("Top Degree",           degree),
        ("Top Closeness",        closeness),
        ("Top Node Betweenness", betweenness),
    ]

    for title, data in sections:
        section(title)
        for node, value in sorted(data.items(), key=lambda x: x[1], reverse=True)[:TOP_N]:
            print(f"  {node:<20} {value:.4f}")

    section("Top Edge Betweenness")
    for (u, v), value in sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)[:TOP_N]:
        print(f"  {u:<20} -- {v:<20} {value:.4f}")


def describe_matrices(G, subgraph_size=15):
    top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n, weight='weight'), reverse=True)[:subgraph_size]
    S = G.subgraph(top_nodes)
    nodes = list(S.nodes())

    print("\nNodes in subgraph:")
    print("  " + ", ".join(nodes))

    section("Adjacency Matrix")
    A = nx.to_numpy_array(S, nodelist=nodes)
    header = "".join(f"{n:>8}" for n in nodes)
    print(f"{'':>20}{header}")
    for i, row in enumerate(A):
        vals = "".join(f"{int(v):>8}" for v in row)
        print(f"  {nodes[i]:<18}{vals}")

    section("Incidence Matrix")
    edges = list(S.edges())
    edge_labels = [f"{u[:6]}-{v[:6]}" for u, v in edges]
    header = "".join(f"{e:>14}" for e in edge_labels)
    print(f"{'':>20}{header}")
    for node in nodes:
        vals = "".join(
            f"{1:>14}" if (node == u or node == v) else f"{0:>14}"
            for u, v in edges
        )
        print(f"  {node:<18}{vals}")


if __name__ == '__main__':
    G = load_graph(os.path.join(DATA, 'connections.csv'))
    draw_graph(G, os.path.join(HERE, 'graph.png'))

    section("Grpah type")
    describe_graph_type(G)

    section("Graph size")
    describe_graph_size(G)

    section("")
    describe_node_centrality(G)

    describe_edge_centrality(G)

    describe_top(G)

    describe_matrices(G)
