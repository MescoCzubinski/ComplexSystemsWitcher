import csv
import os
import networkx as nx
import numpy as np

MATRIX_SUBGRAPH_SIZE = 15

def load_graph(csv_path):
    G = nx.Graph()
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            w = int(row['weight'])
            G.add_edge(row['character1'], row['character2'], weight=w)
    return G


def space(title):
    if title != "":
        title = f" {title} "
    else:
        title = "====="
    print(f"\n==========================={title}===========================\n")


def describe_density(G):
    density = nx.density(G)
    print(f"Density: {density:.6f}")


def describe_diameter(G):
    print(f"Diameter: {nx.diameter(G)}")


def describe_avg_path_length(G):
    print(f"Average path length: {nx.average_shortest_path_length(G):.4f}")

if __name__ == '__main__':
    G = load_graph(os.path.join('data', 'connections.csv'))
    G_cc = G.subgraph(max(nx.connected_components(G), key=len))

    space("Grpah density")
    describe_density(G)

    space("Grpah diameter")
    describe_diameter(G_cc)

    space("Grpah average path length")
    describe_avg_path_length(G_cc)
