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


def esu(adj, order, k):
    out = []

    def extend(sub, sub_set, ext, root):
        if len(sub) == k:
            out.append(sub)
            return
        ext = set(ext)
        while ext:
            w = ext.pop()
            new = {u for u in adj[w]
                   if order[u] > order[root] and u not in sub_set and not (adj[u] & sub_set)}
            extend(sub + [w], sub_set | {w}, ext | new, root)

    for v in adj:
        ext = {u for u in adj[v] if order[u] > order[v]}
        extend([v], {v}, ext, v)
    return out


def classify(nodes, adj):
    deg = [sum(1 for b in nodes if b in adj[a]) for a in nodes]
    return (sum(deg) // 2, tuple(sorted(deg)))


def count_motifs4(G):
    adj = {v: set(G.neighbors(v)) for v in G.nodes()}
    order = {v: i for i, v in enumerate(G.nodes())}
    counts = {name: 0 for name in ORDER4}
    for sub in esu(adj, order, 4):
        name = MOTIF4.get(classify(sub, adj))
        if name:
            counts[name] += 1
    return counts


def significance(real, ensemble):
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
    return res


def add_significance_profile(sig):
    norm = np.sqrt(sum(d['z'] ** 2 for d in sig.values())) or 1.0
    for d in sig.values():
        d['sp'] = d['z'] / norm
    return sig


def describe_significance(sig):
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
        lcc = max((len(c) for c in nx.connected_components(H)), default=0) / n0
        rows.append({'frac': f, 'strategy': strategy,
                     'triangle': m3['triangle'], 'path': m3['path'], 'lcc': lcc})
    return rows


def describe_resilience(rows, fractions):
    by_frac = {f: {} for f in fractions}
    for r in rows:
        by_frac[r['frac']][r['strategy']] = r['triangle']
    print(f"{'removed %':>10}{'targeted':>14}{'random':>14}")
    for f in fractions:
        print(f"{100 * f:>10.0f}{by_frac[f]['targeted']:>14}{by_frac[f]['random']:>14}")


def style_axis(ax, title, xlabel, ylabel):
    ax.set_facecolor(BG)
    ax.set_title(title, color='white', fontsize=14, pad=10)
    ax.set_xlabel(xlabel, color='white', fontsize=10)
    ax.set_ylabel(ylabel, color='white', fontsize=10)
    ax.tick_params(colors='white')
    ax.grid(True, color='white', alpha=0.1, linestyle='--')
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')


def style_legend(ax):
    legend = ax.legend(facecolor=BG, edgecolor='#444444', labelcolor='white')
    if legend and legend.get_title():
        legend.get_title().set_color('white')


def plot_motif_frequency(named_counts, order, title, img_path):
    rows = []
    for name, counts in named_counts:
        total = sum(counts.values()) or 1
        for motif in order:
            rows.append({'network': name, 'motif': motif,
                         'frequency': 100 * counts[motif] / total})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
    sns.barplot(data=df, x='motif', y='frequency', hue='network',
                order=order, ax=ax, palette='Set2', alpha=0.9)
    style_axis(ax, title, 'motif', 'relative frequency [%]')
    ax.tick_params(axis='x', labelrotation=20)
    style_legend(ax)

    plt.tight_layout()
    plt.savefig(img_path, dpi=150, facecolor=BG, bbox_inches='tight')
    plt.close()
    print(f"Saved: {img_path}")


def plot_significance_profile(sig, img_path):
    motifs = list(sig.keys())
    sp = [sig[m]['sp'] for m in motifs]
    colors = ['#66c2a5' if v >= 0 else '#fc8d62' for v in sp]

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
    ax.bar(motifs, sp, color=colors, alpha=0.9)
    ax.axhline(0, color='white', linewidth=0.8, alpha=0.6)
    style_axis(ax, 'Significance Profile (degree-preserving null model)',
               'motif', 'normalised Z-score (SP)')
    ax.tick_params(axis='x', labelrotation=20)

    plt.tight_layout()
    plt.savefig(img_path, dpi=150, facecolor=BG, bbox_inches='tight')
    plt.close()
    print(f"Saved: {img_path}")


def plot_resilience(rows, img_path):
    df = pd.DataFrame(rows)
    df['frac'] = (df['frac'] * 100).round().astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
    for ax, col, label in [(axes[0], 'triangle', 'triangle count'),
                           (axes[1], 'lcc', 'largest component fraction')]:
        for strategy, sub in df.groupby('strategy'):
            ax.plot(sub['frac'], sub[col], marker='o', label=strategy)
        style_axis(ax, f'resilience - {label}', 'nodes removed [%]', label)
        style_legend(ax)

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
    named3 = [('Original', real3), ('Barabási-Albert', ba3)]
    named4 = [('Original', real4), ('Barabási-Albert', ba4)]

    section("3-node motif frequencies")
    describe_frequencies(named3, ORDER3)

    section("4-node motif frequencies")
    describe_frequencies(named4, ORDER4)

    ens3 = [count_motifs3(degree_preserving(G, 37 + i)) for i in range(500)]
    ens4 = [count_motifs4(degree_preserving(G, 37 + i)) for i in range(15)]
    sig3 = add_significance_profile(significance(real3, ens3))
    sig4 = add_significance_profile(significance(real4, ens4))

    section("3-node motif significance")
    describe_significance(sig3)

    section("4-node motif significance")
    describe_significance(sig4)

    fractions = np.linspace(0.0, 0.6, 13)
    rows = resilience(G, fractions, 'targeted', 37) + resilience(G, fractions, 'random', 37)

    section("resilience - triangle count vs node removal")
    describe_resilience(rows, fractions)

    plot_motif_frequency(named3, ORDER3, '3-node motif frequency', os.path.join(HERE, 'motif_freq_3.png'))
    plot_motif_frequency(named4, ORDER4, '4-node motif frequency', os.path.join(HERE, 'motif_freq_4.png'))
    plot_significance_profile(sig3, os.path.join(HERE, 'significance_profile_3.png'))
    plot_significance_profile(sig4, os.path.join(HERE, 'significance_profile_4.png'))
    plot_resilience(rows, os.path.join(HERE, 'resilience.png'))
