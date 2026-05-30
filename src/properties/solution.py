import csv
import os
import sys
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from utils import load_graph, section, DATA, TOP_N, print_stats


def describe_density(G):
    density = nx.density(G)
    print(f"Density: {density:.6f}")


def describe_diameter(G):
    print(f"Diameter: {nx.diameter(G)}")


def describe_avg_path_length(G):
    print(f"Average path length: {nx.average_shortest_path_length(G):.4f}")


def describe_shortest_path(G, source, target):
    if source not in G:
        print(f"Node '{source}' not found.")
        return
    if target not in G:
        print(f"Node '{target}' not found.")
        return

    path = nx.shortest_path(G, source=source, target=target)
    print(f"Path:   {' -> '.join(path)}")


def describe_path_length_distribution(G):
    lengths = []
    for source, targets in nx.shortest_path_length(G):
        for target, length in targets.items():
            if source != target:
                lengths.append(length)

    counter = {}
    for l in lengths:
        counter[l] = counter.get(l, 0) + 1

    print(f"{'Length':>8} {'Count':>10} {'%':>8}")
    print("-" * 36)
    total = len(lengths)
    for length in sorted(counter.keys()):
        count = counter[length]
        pct = 100 * count/total
        print(f"{length:>8} {count:>10} {pct:>7.2f}%")


def describe_centrality_distributions(G):
    degree          = nx.degree_centrality(G)
    closeness       = nx.closeness_centrality(G, distance='weight')
    betweenness     = nx.betweenness_centrality(G, normalized=True)
    edge_betweenness = nx.edge_betweenness_centrality(G, normalized=True)

    node_sections = [
        ("graph centrality distributions - degree",           degree),
        ("graph centrality distributions - closeness",        closeness),
        ("graph centrality distributions - node betweenness", betweenness),
    ]

    for title, data in node_sections:
        values = list(data.values())
        section(title)
        print_stats(values)

    values = list(edge_betweenness.values())
    section("graph centrality distributions - edge betweenness")
    print_stats(values)


def describe_pagerank(G):
    page_rank = nx.pagerank(G, weight='weight')

    print(f"{'Node':<23} {'PageRank':>10}")
    print("-" * 36)
    for node, value in sorted(page_rank.items(), key=lambda x: x[1], reverse=True)[:TOP_N]:
        print(f"  {node:<23} {value:.6f}")

    return page_rank


def describe_connected_components(G):
    components = list(nx.connected_components(G))
    components_sorted = sorted(components, key=len, reverse=True)

    print(f"\n{'#':>4} {'Size':>6}  Nodes")
    print("-" * 36)
    for i, comp in enumerate(components_sorted):
        nodes_preview = ', '.join(sorted(comp)[:3])
        if len(comp) > 3:
            nodes_preview += f', ... (+{len(comp) - 3} more)'
        print(f"{i+1:>4} {len(comp):>6}  {nodes_preview}")


def describe_k_connectivity(G):
    node_connectivity = nx.node_connectivity(G)
    edge_connectivity = nx.edge_connectivity(G)

    print(f"Node connectivity: {node_connectivity}")
    print(f"Edge connectivity: {edge_connectivity}")


def describe_cliques(G, n=4):
    all_cliques = list(nx.find_cliques(G))

    max_clique = max(all_cliques, key=len)
    print(f"Maximum clique (size {len(max_clique)}):")
    print(f"  {', '.join(max_clique)}")

    cliques_n = [c for c in all_cliques if len(c) == n]
    print(f"\nCliques of order {n}: {len(cliques_n)} found")
    for c in cliques_n[:5]:
        print(f"  {', '.join(c)}")

    near_cliques = [c for c in all_cliques if len(c) >= n - 1 and len(c) < n]
    print(f"\nNear-cliques (size={n-1}): {len(near_cliques)} found")
    for c in near_cliques[:5]:
        print(f"  {', '.join(c)}")


if __name__ == '__main__':
    G = load_graph(os.path.join(DATA, 'connections.csv'))
    G_cc = G.subgraph(max(nx.connected_components(G), key=len))

    section("graph density")
    describe_density(G)

    section("graph diameter")
    describe_diameter(G_cc)

    section("graph average path length")
    describe_avg_path_length(G_cc)

    section("graph average path length")
    describe_shortest_path(G_cc, "Nenneke", "Emhyr var Emreis")

    section("graph path length distribution")
    describe_path_length_distribution(G_cc)

    describe_centrality_distributions(G_cc)

    section("graph PageRank")
    describe_pagerank(G_cc)

    section("graph connected components")
    describe_connected_components(G)

    section("graph connectivity")
    describe_k_connectivity(G_cc)

    section("graph cliques")
    describe_cliques(G_cc)
