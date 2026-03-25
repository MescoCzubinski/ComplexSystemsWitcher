import csv
import os
import networkx as nx
import numpy as np

DATA_DIR = 'data'
MIN_WEIGHT = 5
TOP_N = 10
MATRIX_SUBGRAPH_SIZE = 15


def load_graph(csv_path):
    G = nx.Graph()
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            w = int(row['weight'])
            if w >= MIN_WEIGHT:
                G.add_edge(row['name1'], row['name2'], weight=w)
    return G


def section(title):
    print(f'\n{"═" * 60}')
    print(f'  {title}')
    print(f'{"═" * 60}')


def top_table(label, scores, n=TOP_N):
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    print(f'\n  Top {n} — {label}:')
    for i, (name, val) in enumerate(ranked[:n], 1):
        print(f'    {i:>2}. {name:<40}  {val:.6f}')
    return ranked


def print_matrix(matrix, labels, title):
    print(f'\n  {title}')
    col_w = max(len(l) for l in labels) + 2
    header = ' ' * col_w + '  '.join(f'{l:>{col_w}}' for l in labels)
    print('  ' + header)
    for i, row_label in enumerate(labels):
        row = '  '.join(f'{int(matrix[i, j]):>{col_w}}' for j in range(len(labels)))
        print(f'  {row_label:>{col_w}}  {row}')


if __name__ == '__main__':
    G = load_graph(os.path.join(DATA_DIR, 'connections.csv'))

    # ── 1. Rząd i rozmiar ────────────────────────────────────────────────────
    section('1. Rząd i rozmiar grafu')
    print(f'  Rząd  (liczba wierzchołków): {G.number_of_nodes()}')
    print(f'  Rozmiar (liczba krawędzi):   {G.number_of_edges()}')

    # ── 2. Centralności wierzchołków ─────────────────────────────────────────
    section('2. Miary centralności wierzchołków')

    degree_c     = nx.degree_centrality(G)
    closeness_c  = nx.closeness_centrality(G, distance='weight')
    betweenness_c = nx.betweenness_centrality(G, weight='weight', normalized=True)

    top_deg = top_table('Stopień (degree centrality)',      degree_c)
    top_clo = top_table('Bliskość (closeness centrality)',  closeness_c)
    top_bet = top_table('Pośrednictwo (betweenness centrality)', betweenness_c)

    print(f'\n  Wszystkie wartości (sortowane malejąco po stopniu):')
    all_nodes = sorted(degree_c, key=lambda n: -degree_c[n])
    print(f'  {"Wierzchołek":<40} {"Stopień":>10} {"Bliskość":>10} {"Pośrednictwo":>14}')
    print(f'  {"-"*40} {"-"*10} {"-"*10} {"-"*14}')
    for n in all_nodes:
        print(f'  {n:<40} {degree_c[n]:>10.4f} {closeness_c[n]:>10.4f} {betweenness_c[n]:>14.6f}')

    # ── 3. Centralność krawędzi ───────────────────────────────────────────────
    section('3. Pośrednictwo krawędzi (edge betweenness centrality)')

    edge_bet = nx.edge_betweenness_centrality(G, weight='weight', normalized=True)

    print(f'\n  Top {TOP_N} krawędzi:')
    ranked_edges = sorted(edge_bet.items(), key=lambda x: -x[1])
    for i, ((u, v), val) in enumerate(ranked_edges[:TOP_N], 1):
        print(f'    {i:>2}. {u} — {v:<50}  {val:.6f}')

    print(f'\n  Wszystkie krawędzie (sortowane malejąco):')
    print(f'  {"Krawędź":<80} {"Pośrednictwo":>14}')
    print(f'  {"-"*80} {"-"*14}')
    for (u, v), val in ranked_edges:
        edge_str = f'{u} — {v}'
        print(f'  {edge_str:<80} {val:>14.6f}')

    # ── 4. Najważniejsze wierzchołki i krawędzie ─────────────────────────────
    section('4. Najważniejsze wierzchołki i krawędzie')

    print(f'\n  Najważniejszy wierzchołek wg stopnia:      {top_deg[0][0]}')
    print(f'  Najważniejszy wierzchołek wg bliskości:    {top_clo[0][0]}')
    print(f'  Najważniejszy wierzchołek wg pośrednictwa: {top_bet[0][0]}')
    print(f'\n  Najważniejsza krawędź wg pośrednictwa:     {ranked_edges[0][0][0]} — {ranked_edges[0][0][1]}')

    # ── 5. Macierze ───────────────────────────────────────────────────────────
    section(f'5. Macierz sąsiedztwa i incydencji  (podgraf: top {MATRIX_SUBGRAPH_SIZE} wg stopnia)')

    top_nodes = [n for n, _ in top_deg[:MATRIX_SUBGRAPH_SIZE]]
    SG = G.subgraph(top_nodes)
    sg_nodes = list(SG.nodes())
    sg_edges = list(SG.edges())

    # Adjacency matrix
    adj = nx.to_numpy_array(SG, nodelist=sg_nodes)
    print_matrix(adj.astype(int), sg_nodes, 'Macierz sąsiedztwa (A):')

    # Incidence matrix
    inc = nx.incidence_matrix(SG, nodelist=sg_nodes, edgelist=sg_edges).toarray().astype(int)
    edge_labels = [f'{u[:6]}–{v[:6]}' for u, v in sg_edges]
    print(f'\n  Macierz incydencji (B)  [{len(sg_nodes)} wierzchołków × {len(sg_edges)} krawędzi]:')
    col_w = 14
    header = ' ' * 42 + '  '.join(f'{l:>{col_w}}' for l in edge_labels)
    print('  ' + header)
    for i, node in enumerate(sg_nodes):
        row = '  '.join(f'{inc[i, j]:>{col_w}}' for j in range(len(sg_edges)))
        print(f'  {node:<40}  {row}')
