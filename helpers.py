import csv
import networkx as nx


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
