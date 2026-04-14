import csv
import os
import networkx as nx
import numpy as np
from helpers import load_graph, section


MATRIX_SUBGRAPH_SIZE = 15
TOP_N = 10

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
    G = load_graph(os.path.join('data', 'connections.csv'))

    section("Grpah type")
    describe_graph_type(G)

    section("Graph size")
    describe_graph_size(G)

    section("")
    describe_node_centrality(G)

    describe_edge_centrality(G)

    describe_top(G)

    describe_matrices(G)
