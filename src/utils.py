import csv
import os
import networkx as nx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')


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
