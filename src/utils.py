import csv
import os
import matplotlib
import networkx as nx
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')

BG = '#1c1c1c'
TOP_N = 10


def load_graph(csv_path):
    G = nx.Graph()
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            w = int(row['weight'])
            G.add_edge(row['character1'], row['character2'], weight=w)
    return G


def section(title):
    if title != "":
        title = f" {title} "
    else:
        title = "====="
    print(f"\n==========================={title}===========================\n")


def node_color_map(communities):
    cmap = matplotlib.colormaps['hsv'].resampled(len(communities))
    return {node: cmap(i) for i, comm in enumerate(communities) for node in comm}


def print_stats(values):
    print(f"  Min:    {min(values):.4f}")
    print(f"  Max:    {max(values):.4f}")
    print(f"  Mean:   {np.mean(values):.4f}")
    print(f"  Median: {np.median(values):.4f}")
    print(f"  Std:    {np.std(values):.4f}")
