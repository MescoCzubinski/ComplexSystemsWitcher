import os
import random
import sys
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from utils import load_graph, section, DATA, BG

os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)

MOTIF3 = {
    (2, (1, 1, 2)): 'path',
    (3, (2, 2, 2)): 'triangle',
}
MOTIF4 = {
    (3, (1, 1, 2, 2)): 'path',
    (3, (1, 1, 1, 3)): 'star',
    (4, (2, 2, 2, 2)): 'cycle',
    (4, (1, 2, 2, 3)): 'triangle+tail',
    (5, (2, 2, 3, 3)): 'diamond',
    (6, (3, 3, 3, 3)): 'clique',
}
ORDER3 = list(MOTIF3.values())
ORDER4 = list(MOTIF4.values())


def generate_ba(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    k = max(1, round(m / n))
    R = nx.barabasi_albert_graph(n, k)
    for _, _, d in R.edges(data=True):
        d['weight'] = 1
    return R


def degree_preserving(G, seed):
    R = G.copy()
    m = R.number_of_edges()
    try:
        nx.double_edge_swap(R, nswap=10 * m, max_tries=100 * m, seed=seed)
    except nx.NetworkXError:
        pass
    return R


def count_motifs3(G):
    triangles = sum(nx.triangles(G).values()) // 3
    wedges = sum(d * (d - 1) // 2 for _, d in G.degree())
    paths = wedges - 3 * triangles
    return {'path': paths, 'triangle': triangles}


def esu(adj, order, k, probs, rng):
    out = []

    def extend(sub, sub_set, ext, root):
        s = len(sub)
        p = probs[s]
        ext = set(ext)
        if s + 1 == k:
            while ext:
                w = ext.pop()
                if p >= 1.0 or rng.random() < p:
                    out.append(sub + [w])
            return
        while ext:
            w = ext.pop()
            if p < 1.0 and rng.random() >= p:
                continue
            new = {u for u in adj[w]
                   if order[u] > order[root] and u not in sub_set and not (adj[u] & sub_set)}
            extend(sub + [w], sub_set | {w}, ext | new, root)

    for v in adj:
        if probs[0] < 1.0 and rng.random() >= probs[0]:
            continue
        ext = {u for u in adj[v] if order[u] > order[v]}
        extend([v], {v}, ext, v)
    return out


def classify(nodes, adj):
    deg = [sum(1 for b in nodes if b in adj[a]) for a in nodes]
    return (sum(deg) // 2, tuple(sorted(deg)))


def count_motifs4(G, probs=(1, 1, 1, 1), rng=random):
    adj = {v: set(G.neighbors(v)) for v in G.nodes()}
    order = {v: i for i, v in enumerate(G.nodes())}
    factor = 1.0 / (probs[0] * probs[1] * probs[2] * probs[3])
    counts = {name: 0.0 for name in ORDER4}
    for sub in esu(adj, order, 4, probs, rng):
        name = MOTIF4.get(classify(sub, adj))
        if name:
            counts[name] += factor
    return {name: round(c) for name, c in counts.items()}


def significance(G, real, count_fn, n_samples, seed=37):
    ensemble = [count_fn(degree_preserving(G, seed + i)) for i in range(n_samples)]
    res = {}
    for name, rc in real.items():
        vals = np.array([e[name] for e in ensemble], dtype=float)
        mu, sd = vals.mean(), vals.std()
        z = (rc - mu) / sd if sd > 0 else 0.0
        if rc >= mu:
            p = (1 + np.sum(vals >= rc)) / (len(vals) + 1)
        else:
            p = (1 + np.sum(vals <= rc)) / (len(vals) + 1)
        res[name] = {'real': rc, 'mean': mu, 'std': sd, 'z': z, 'p': p}

    norm = np.sqrt(sum(d['z'] ** 2 for d in res.values())) or 1.0
    for d in res.values():
        d['sp'] = d['z'] / norm
    return res


def describe_significance(G, real, count_fn, n_samples):
    sig = significance(G, real, count_fn, n_samples)
    print(f"{'motif':<22}{'real':>10}{'rand mean':>12}{'std':>10}{'Z':>9}{'p':>9}{'SP':>9}")
    for motif, d in sig.items():
        print(f"{motif:<22}{d['real']:>10}{d['mean']:>12.1f}{d['std']:>10.2f}"
              f"{d['z']:>9.2f}{d['p']:>9.3f}{d['sp']:>9.3f}")


def describe_frequencies(named_counts, order):
    print(f"{'motif':<22}" + "".join(f"{n:>20}" for n, _ in named_counts))
    for motif in order:
        cells = ""
        for _, counts in named_counts:
            total = sum(counts.values()) or 1
            cells += f"{counts[motif]:>10} ({100 * counts[motif] / total:>5.1f}%)"
        print(f"{motif:<22}{cells}")


def resilience(G, fractions, strategy, seed):
    n0 = G.number_of_nodes()
    if strategy == 'targeted':
        order = [n for n, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)]
    else:
        order = list(G.nodes())
        random.Random(seed).shuffle(order)

    rows = []
    for f in fractions:
        H = G.copy()
        H.remove_nodes_from(order[:int(f * n0)])
        m3 = count_motifs3(H)
        rows.append({'frac': f, 'strategy': strategy, 'triangle': m3['triangle'], 'path': m3['path']})
    return rows


def describe_resilience(G, seed=37):
    fractions = np.linspace(0.0, 0.6, 13)
    rows = resilience(G, fractions, 'targeted', seed) + resilience(G, fractions, 'random', seed)
    by_frac = {f: {} for f in fractions}
    for r in rows:
        by_frac[r['frac']][r['strategy']] = r
    print(f"{'removed %':>10}{'tri targ':>12}{'tri rand':>12}{'path targ':>12}{'path rand':>12}")
    for f in fractions:
        t, rnd = by_frac[f]['targeted'], by_frac[f]['random']
        print(f"{100 * f:>10.0f}{t['triangle']:>12}{rnd['triangle']:>12}"
              f"{t['path']:>12}{rnd['path']:>12}")


def plot_motif_frequency(named_counts, order, title, img_path):
    rows = []
    for name, counts in named_counts:
        total = sum(counts.values()) or 1
        for motif in order:
            rows.append({'network': name, 'motif': motif,
                         'frequency': 100 * counts[motif] / total})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
    ax.set_facecolor(BG)

    sns.barplot(data=df, x='motif', y='frequency', hue='network',
                order=order, ax=ax, palette='Set2', alpha=0.9)

    ax.set_title(title, color='white', fontsize=14, pad=10)
    ax.set_xlabel('motif', color='white', fontsize=10)
    ax.set_ylabel('relative frequency [%]', color='white', fontsize=10)
    ax.tick_params(colors='white')
    ax.tick_params(axis='x', labelrotation=20)
    ax.grid(True, color='white', alpha=0.1, linestyle='--')
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

    legend = ax.legend(facecolor=BG, edgecolor='#444444', labelcolor='white')
    if legend and legend.get_title():
        legend.get_title().set_color('white')

    plt.tight_layout()
    plt.savefig(img_path, dpi=150, facecolor=BG, bbox_inches='tight')
    plt.close()
    print(f"Saved: {img_path}")


if __name__ == '__main__':
    random.seed(37)
    np.random.seed(37)

    G = load_graph(os.path.join(DATA, 'connections.csv'))
    G_ba = generate_ba(G)

    real3 = count_motifs3(G)
    real4 = count_motifs4(G)
    ba3 = count_motifs3(G_ba)
    ba4 = count_motifs4(G_ba)

    section("3-node motif frequencies")
    describe_frequencies([('Original', real3), ('Barabási-Albert', ba3)], ORDER3)

    section("4-node motif frequencies")
    describe_frequencies([('Original', real4), ('Barabási-Albert', ba4)], ORDER4)

    section("3-node motif significance")
    describe_significance(G, real3, count_motifs3, 500)

    section("4-node motif significance")
    describe_significance(G, real4, lambda H: count_motifs4(H, probs=(1, 1, 1, 0.1)), 50)

    section("resilience - triangle count vs node removal")
    describe_resilience(G, seed=37)

    plot_motif_frequency([('Original', real3), ('Barabási-Albert', ba3)], ORDER3, '3-node motif frequency', os.path.join(HERE, 'figures/motif_freq_3.png'))
    plot_motif_frequency([('Original', real4), ('Barabási-Albert', ba4)], ORDER4, '4-node motif frequency', os.path.join(HERE, 'figures/motif_freq_4.png'))
