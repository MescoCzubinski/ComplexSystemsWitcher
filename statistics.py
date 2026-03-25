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
            G.add_edge(row['name1'], row['name2'], weight=w)
    return G


def section(title):
    print(f'\n{"═" * 60}')
    print(f'  {title}')


def print_matrix(matrix, labels, title):
    print(f'\n  {title}')
    col_w = max(len(l) for l in labels) + 2
    header = ' ' * col_w + '  '.join(f'{l:>{col_w}}' for l in labels)
    print('  ' + header)
    for i, row_label in enumerate(labels):
        row = '  '.join(f'{int(matrix[i, j]):>{col_w}}' for j in range(len(labels)))
        print(f'  {row_label:>{col_w}}  {row}')



def get_order_and_size(G):
    return {'nodes': G.number_of_nodes(), 'edges': G.number_of_edges()}

def print_order_and_size(data):
    section('1. Rząd i rozmiar grafu')
    print(f'  Rząd  (liczba wierzchołków): {data["nodes"]}')
    print(f'  Rozmiar (liczba krawędzi):   {data["edges"]}')

def save_order_and_size(data, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['nodes', data['nodes']])
        writer.writerow(['edges', data['edges']])


def get_node_centrality(G):
    degree_c      = nx.degree_centrality(G)
    closeness_c   = nx.closeness_centrality(G, distance='weight')
    betweenness_c = nx.betweenness_centrality(G, weight='weight', normalized=True)
    return degree_c, closeness_c, betweenness_c

def print_node_centrality(degree_c, closeness_c, betweenness_c):
    section('2. Miary centralności wierzchołków')
    top_deg = sorted(degree_c.items(), key=lambda x: -x[1])
    print(f'\n  {"Wierzchołek":<40} {"Stopień":>10} {"Bliskość":>10} {"Pośrednictwo":>14}')
    print(f'  {"-"*40} {"-"*10} {"-"*10} {"-"*14}')
    for node, _ in top_deg:
        print(f'  {node:<40} {degree_c[node]:>10.4f} {closeness_c[node]:>10.4f} {betweenness_c[node]:>14.6f}')

def save_node_centrality(degree_c, closeness_c, betweenness_c, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['node', 'degree', 'closeness', 'betweenness'])
        for node, _ in sorted(degree_c.items(), key=lambda x: -x[1]):
            writer.writerow([node, degree_c[node], closeness_c[node], betweenness_c[node]])


def get_edge_centrality(G):
    edge_bet = nx.edge_betweenness_centrality(G, weight='weight', normalized=True)
    return sorted(edge_bet.items(), key=lambda x: -x[1])

def print_edge_centrality(ranked_edges):
    section('3. Pośrednictwo krawędzi (edge betweenness centrality)')
    print(f'\n  {"Krawędź":<80} {"Pośrednictwo":>14}')
    print(f'  {"-"*80} {"-"*14}')
    for (u, v), val in ranked_edges:
        print(f'  {u} — {v:<{78 - len(u)}} {val:>14.6f}')

def save_edge_centrality(ranked_edges, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['node1', 'node2', 'betweenness'])
        for (u, v), val in ranked_edges:
            writer.writerow([u, v, val])


def get_most_important(degree_c, closeness_c, betweenness_c, ranked_edges):
    return {
        'top_deg': sorted(degree_c.items(), key=lambda x: -x[1])[0],
        'top_clo': sorted(closeness_c.items(), key=lambda x: -x[1])[0],
        'top_bet': sorted(betweenness_c.items(), key=lambda x: -x[1])[0],
        'top_edge': ranked_edges[0],
    }

def print_most_important(data):
    section('4. Najważniejsze wierzchołki i krawędzie')
    print(f'\n  Najważniejszy wierzchołek wg stopnia:      {data["top_deg"][0]}')
    print(f'  Najważniejszy wierzchołek wg bliskości:    {data["top_clo"][0]}')
    print(f'  Najważniejszy wierzchołek wg pośrednictwa: {data["top_bet"][0]}')
    (u, v), _ = data['top_edge']
    print(f'\n  Najważniejsza krawędź wg pośrednictwa:     {u} — {v}')

def save_most_important(data, path):
    (eu, ev), _ = data['top_edge']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['top_node_degree',      data['top_deg'][0]])
        writer.writerow(['top_node_closeness',   data['top_clo'][0]])
        writer.writerow(['top_node_betweenness', data['top_bet'][0]])
        writer.writerow(['top_edge_betweenness', f'{eu} — {ev}'])


def get_matrices(G, degree_c):
    top_nodes = [n for n, _ in sorted(degree_c.items(), key=lambda x: -x[1])[:MATRIX_SUBGRAPH_SIZE]]
    SG = G.subgraph(top_nodes)
    sg_nodes = list(SG.nodes())
    sg_edges = list(SG.edges())
    adj = nx.to_numpy_array(SG, nodelist=sg_nodes)
    inc = nx.incidence_matrix(SG, nodelist=sg_nodes, edgelist=sg_edges).toarray().astype(int)
    return sg_nodes, sg_edges, adj, inc

def print_matrices(sg_nodes, sg_edges, adj, inc):
    section(f'5. Macierz sąsiedztwa i incydencji  (podgraf: top {MATRIX_SUBGRAPH_SIZE} wg stopnia)')
    print_matrix(adj.astype(int), sg_nodes, 'Macierz sąsiedztwa (A):')

    edge_labels = [f'{u[:6]}–{v[:6]}' for u, v in sg_edges]
    print(f'\n  Macierz incydencji (B)  [{len(sg_nodes)} wierzchołków × {len(sg_edges)} krawędzi]:')
    col_w = 14
    header = ' ' * 42 + '  '.join(f'{l:>{col_w}}' for l in edge_labels)
    print('  ' + header)
    for i, node in enumerate(sg_nodes):
        row = '  '.join(f'{inc[i, j]:>{col_w}}' for j in range(len(sg_edges)))
        print(f'  {node:<40}  {row}')

def save_matrices(sg_nodes, sg_edges, adj, inc, adj_path, inc_path):
    with open(adj_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([''] + sg_nodes)
        for i, node in enumerate(sg_nodes):
            writer.writerow([node] + list(adj[i].astype(int)))

    with open(inc_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([''] + [f'{u}—{v}' for u, v in sg_edges])
        for i, node in enumerate(sg_nodes):
            writer.writerow([node] + list(inc[i]))


if __name__ == '__main__':
    G = load_graph(os.path.join('data', 'connections.csv'))

    print_order_and_size(get_order_and_size(G))

    degree_c, closeness_c, betweenness_c = get_node_centrality(G)
    # print_node_centrality(degree_c, closeness_c, betweenness_c)

    ranked_edges = get_edge_centrality(G)
    # print_edge_centrality(ranked_edges)

    print_most_important(get_most_important(degree_c, closeness_c, betweenness_c, ranked_edges))

    # print_matrices(*get_matrices(G, degree_c))

