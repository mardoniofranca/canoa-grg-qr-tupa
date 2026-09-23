"""
Reproduction of the models and figures from:

  H. Jnane, G. Di Molfetta & F. M. Miatto (2020)
  "Growing Random Graphs with Quantum Rules", EPTCS 315, pp. 38-47.

Model (Sections 2.1 and 2.2 of the paper):
  1. Hamiltonian = adjacency matrix A of the graph; U(t) = exp(-i A t).
  2. The walker(s) evolve for a time t ~ Exp(mean tau).
  3. The position of each walker is measured: P(v) = |<v|U(t)|psi>|^2.
  4. A single new node is attached to all measured nodes (1 walker -> trees;
     2 or more -> graphs with cycles; if the walkers collide, the new node
     has a single edge).
  5. Each walker restarts at the node where it collapsed.
  The initial graph is a single node.

Generated figures (OUTPUT_DIR folder):
  fig2_trees_1_walker.png        Fig. 2  (n=100, tau = 0.001 ... 10)
  fig3_stars_spectrum.png        Fig. 3  (connected stars + eigenvalues)
  fig4_graphs_2_walkers.png      Fig. 4  (n=100, tau = 0.001 ... 10)
  fig5_degree_distribution.png   Fig. 5  (degree distribution, 1/2/3 walkers)
  fig6_diameter.png              Fig. 6  (diameter vs. tau)
  fig7_leaf_fraction.png         Fig. 7  (leaf fraction vs. tau, 1 walker)
  fig8_clustering.png            Fig. 8  (clustering vs. tau)
  extra_star_size.png            Check of E[n] ~ 1/tau (Section 2.1)

Usage (terminal):
  python reproduce_jnane_paper.py              # default configuration
  python reproduce_jnane_paper.py --fast       # lightweight version (a few minutes)
  python reproduce_jnane_paper.py --samples 20
  python reproduce_jnane_paper.py --reuse      # only rebuild the figures (uses results.pkl)

Usage (Jupyter): paste/run the file in a cell and then call, e.g.,
  main(fast=True)   or   main()   or   main(samples=20, reuse=False)

Dependencies: numpy, scipy, networkx, matplotlib
"""
import argparse
import os
import pickle
import time

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from scipy.sparse.csgraph import shortest_path

OUTPUT_DIR = "paper_figures"
TAUS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 3.0, 10.0]
TAUS_DEGREE = [0.1, 0.5, 1.0, 10.0]                  # Fig. 5
TAUS_SAMPLE = [0.001, 0.01, 0.05, 0.1, 0.5, 10.0]    # Figs. 2 and 4
WALKER_COUNTS = [1, 2, 3]


# =====================================================================
#  Model: growth driven by a continuous-time quantum walk
# =====================================================================
def _distribution(A, n, edges, positions, t, max_degree):
    """P(v) for each walker after evolving for a time t (n x n_walkers matrix).
    Automatically selects the cheaper method; both are exact."""
    w = len(positions)
    sparse_cost = 4e-5 * t * max_degree + 5e-4          # empirical estimate (s)
    dense_cost = 3.5e-2 * (n / 600) ** 3

    if sparse_cost < dense_cost:                        # sparse route
        row, col = zip(*edges)
        M = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
        M = M + M.T
        psi0 = np.zeros((n, w), dtype=complex)
        psi0[positions, np.arange(w)] = 1.0
        psi = expm_multiply(-1j * t * M, psi0)
    else:                                                # dense diagonalization
        eigval, V = np.linalg.eigh(A[:n, :n])
        psi = V @ (np.exp(-1j * eigval * t)[:, None] * V[positions, :].T)

    p = np.abs(psi) ** 2
    return p / p.sum(axis=0, keepdims=True)


def compute_metrics(A, n):
    """Metrics of a graph with n nodes (dense adjacency matrix)."""
    B = A[:n, :n]
    degrees = B.sum(axis=1).astype(int)
    tri = np.einsum("ij,ji->i", B @ B, B)                # (A^3)_ii = 2 x triangles
    denom = degrees * (degrees - 1)
    Ci = np.divide(tri, denom, out=np.zeros(n), where=denom > 0)
    dist = shortest_path(sp.csr_matrix(B), unweighted=True, directed=False)
    return dict(degrees=degrees,
                leaf_fraction=float(np.mean(degrees == 1)),
                clustering=float(Ci.mean()),
                diameter=float(dist.max()))


def grow_graph(n_final, tau, n_walkers=1, seed=None, checkpoints=(),
               record_spectrum=False):
    rng = np.random.default_rng(seed)
    A = np.zeros((n_final, n_final))
    degrees = np.zeros(n_final, dtype=int)
    edges = []
    n = 1
    positions = [0] * n_walkers
    walker_path, snapshots, spectrum = [], {}, []

    while n < n_final:
        t = rng.exponential(tau)
        if n == 1:
            new_nodes = [0] * n_walkers
        else:
            P = _distribution(A, n, edges, positions, t, max(1, degrees[:n].max()))
            new_nodes = [int(rng.choice(n, p=P[:, k])) for k in range(n_walkers)]
        for v in set(new_nodes):
            A[n, v] = A[v, n] = 1.0
            degrees[v] += 1
            degrees[n] += 1
            edges.append((v, n))
        walker_path.append(new_nodes[0])
        positions = new_nodes
        n += 1

        if record_spectrum:
            eigval = np.clip(np.linalg.eigvalsh(A[:n, :n])[::-1][:3], 0, None)
            spectrum.append(np.pad(eigval, (0, 3 - len(eigval))))
        if n in checkpoints:
            snapshots[n] = compute_metrics(A, n)

    return dict(A=A, snapshots=snapshots, walker_path=walker_path,
                spectrum=np.array(spectrum))


# =====================================================================
#  Simulation batch (Figs. 5 to 8)
# =====================================================================
def run_batch(n_final, checkpoints, n_samples):
    results = {}              # (n_walkers, tau) -> list of dict(snapshots, walker_path)
    total = len(WALKER_COUNTS) * len(TAUS) * n_samples
    done, t0 = 0, time.time()
    for w in WALKER_COUNTS:
        for tau in TAUS:
            runs = []
            for s in range(n_samples):
                r = grow_graph(n_final, tau, w, seed=[w, TAUS.index(tau), s],
                                checkpoints=checkpoints)
                runs.append(dict(snapshots=r["snapshots"], walker_path=r["walker_path"]))
                done += 1
            results[(w, tau)] = runs
            dt = time.time() - t0
            print(f"  [{done:>4}/{total}] walkers={w} tau={tau:<6g} "
                  f"({dt:5.0f}s elapsed)", flush=True)
    return results


def mean_std(runs, n, key):
    v = np.array([r["snapshots"][n][key] for r in runs], dtype=float)
    return v.mean(), v.std()


# =====================================================================
#  Figures
# =====================================================================
def _layout(G, seed=1):
    return nx.spring_layout(G, seed=seed, iterations=150)


def fig_samples(n_walkers, filename, title, n_nodes=101):
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    for ax, tau in zip(axs.ravel(), TAUS_SAMPLE):
        A = grow_graph(n_nodes, tau, n_walkers, seed=7)["A"]
        G = nx.from_numpy_array(A)
        nx.draw(G, _layout(G), ax=ax, node_size=6, width=0.5, node_color="#1f77b4")
        ax.set_title(f"\u03c4 = {tau:g}", fontsize=10)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=130)
    plt.close(fig)


def fig_stars_spectrum(filename, tau=0.01, n_nodes=300, attempts=40):
    """Searches for a realization with ~3 stars (as in Fig. 3) and plots
    the graph together with the evolution of the 3 largest eigenvalues."""
    best_run, best_score = None, -1
    for seed in range(attempts):
        r = grow_graph(n_nodes, tau, 1, seed=seed, record_spectrum=True)
        final = np.sort(r["spectrum"][-1])
        score = min(final[-3:])                # large 3rd eigenvalue = 3 stars
        if score > best_score:
            best_run, best_score = r, score
    r = best_run
    G = nx.from_numpy_array(r["A"])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    nx.draw(G, _layout(G), ax=a1, node_size=5, width=0.4, node_color="#1f77b4")
    a1.set_title(f"Graph after {n_nodes} nodes (\u03c4 = {tau:g})")
    for i, color in enumerate(["C0", "C1", "C2"]):
        a2.plot(np.arange(2, n_nodes + 1), r["spectrum"][:, i], color=color,
                label=f"eigenvalue {i + 1}")
    a2.set_xlabel("index (number of nodes)")
    a2.set_ylabel("value")
    a2.set_title("Evolution of the 3 largest eigenvalues")
    a2.legend()
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=130)
    plt.close(fig)


def fig_degree_distribution(results, n, filename):
    fig, axs = plt.subplots(3, 1, figsize=(6.5, 13))
    colors = plt.cm.viridis(np.linspace(0.85, 0.05, len(TAUS_DEGREE)))
    kmax = 60
    for ax, w in zip(axs, WALKER_COUNTS):
        for tau, color in zip(TAUS_DEGREE, colors):
            H = np.array([np.bincount(r["snapshots"][n]["degrees"], minlength=kmax + 1)[:kmax + 1] / n
                          for r in results[(w, tau)]])
            m, s = H.mean(0), H.std(0)
            k = np.arange(kmax + 1)
            ok = (k >= 1) & (m > 0)
            ax.plot(k[ok], m[ok], color=color, label=f"{tau:g}")
            ax.fill_between(k[ok], np.clip(m - s, 1e-4, None)[ok], (m + s)[ok],
                            color=color, alpha=0.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1)
        ax.set_xlabel("degree")
        ax.set_ylabel("degree probability")
        ax.set_title(f"Degree distribution \u2014 {w} walker(s) \u2014 {n} nodes")
        ax.legend(title="\u03c4")
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=130)
    plt.close(fig)


def _curve_vs_tau(results, n, key, filename, title, ylabel, logy=False, walkers=WALKER_COUNTS):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for w in walkers:
        M = np.array([mean_std(results[(w, tau)], n, key) for tau in TAUS])
        ax.plot(TAUS, M[:, 0], marker="o", ms=3, label=f"{w} walker(s)")
        ax.fill_between(TAUS, M[:, 0] - M[:, 1], M[:, 0] + M[:, 1], alpha=0.2)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("\u03c4")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=130)
    plt.close(fig)


def fig_leaf_fraction(results, sizes, filename):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.cm.magma(np.linspace(0.75, 0.15, len(sizes)))
    for n, color in zip(sizes, colors):
        M = np.array([mean_std(results[(1, tau)], n, "leaf_fraction") for tau in TAUS]) * 100
        ax.plot(TAUS, M[:, 0], color=color, marker="o", ms=3, label=f"{n} nodes")
        ax.fill_between(TAUS, M[:, 0] - M[:, 1], M[:, 0] + M[:, 1], color=color, alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("\u03c4")
    ax.set_ylabel("leaf fraction (%)")
    ax.set_title("Leaf fraction \u2014 1 walker")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=130)
    plt.close(fig)


def fig_star_size(filename, n_final, n_samples):
    """Checks E[n] ~ 1/tau (Section 2.1). Star size = number of consecutive
    collapses at the same node (the walker only leaves the star by
    collapsing onto an external node). Dedicated, longer simulations are
    used to obtain enough stars per run."""
    taus = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    means = []
    for i, tau in enumerate(taus):
        sizes = []
        for k in range(n_samples):
            path = np.array(grow_graph(n_final, tau, 1, seed=[99, i, k])["walker_path"])
            cut = np.flatnonzero(np.diff(path) != 0) + 1
            runs = np.diff(np.concatenate([[0], cut, [len(path)]]))[:-1]  # drop the last, incomplete run
            sizes.extend(runs)
        means.append(np.mean(sizes))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(taus, means, "o-", label="simulation (mean star size)")
    ax.loglog(taus, 1 / np.array(taus), "k--", label="1/\u03c4 (paper)")
    ax.set_xlabel("\u03c4")
    ax.set_ylabel("E[n]")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=130)
    plt.close(fig)
    return taus, means


# =====================================================================
def main(fast=False, samples=None, reuse=False):
    """Run everything. In Jupyter/IPython, call this directly, e.g.:
         main(fast=True)        # lightweight version
         main(samples=20)       # smoother curves
         main(reuse=True)       # only rebuild the figures
    """
    if fast:
        n_final, checkpoints, n_samples = 300, (75, 150, 225, 300), samples or 4
        n_degree = 300
    else:
        n_final, checkpoints, n_samples = 600, (150, 300, 450, 500, 600), samples or 8
        n_degree = 500

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_path = os.path.join(OUTPUT_DIR, "results.pkl")

    print("Figs. 2, 3 and 4 (sample graphs)...")
    fig_samples(1, "fig2_trees_1_walker.png",
                "1 walker \u2014 100 steps \u2014 growth into trees")
    fig_samples(2, "fig4_graphs_2_walkers.png",
                "2 walkers \u2014 100 steps \u2014 graphs with cycles")
    fig_stars_spectrum("fig3_stars_spectrum.png")

    if reuse and os.path.exists(results_path):
        print("Loading", results_path)
        with open(results_path, "rb") as f:
            results = pickle.load(f)
    else:
        print(f"Simulation batch: {n_final} nodes, {n_samples} samples per point...")
        results = run_batch(n_final, checkpoints, n_samples)
        with open(results_path, "wb") as f:
            pickle.dump(results, f)

    print("Figs. 5 to 8...")
    fig_degree_distribution(results, n_degree, "fig5_degree_distribution.png")
    _curve_vs_tau(results, n_degree, "diameter", "fig6_diameter.png",
                  f"Diameter \u2014 {n_degree} nodes", "diameter", logy=True)
    fig_leaf_fraction(results, [c for c in checkpoints if c != 500], "fig7_leaf_fraction.png")
    _curve_vs_tau(results, n_degree, "clustering", "fig8_clustering.png",
                  f"Clustering coefficient \u2014 {n_degree} nodes", "clustering")

    print("Checking E[n] ~ 1/tau...")
    taus, means = fig_star_size("extra_star_size.png",
                                600 if fast else 1500, n_samples)
    print("\nMean star size (simulation) vs. 1/tau:")
    for t, m in zip(taus, means):
        print(f"  tau={t:<6g} E[n]={m:7.1f}   1/tau={1 / t:7.1f}   ratio={m * t:.2f}")
    print(f"\nFigures in ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="fewer nodes and samples")
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--reuse", action="store_true", help="reuse results.pkl")
    # parse_known_args ignores extra arguments (e.g. '-f kernel.json' injected by Jupyter)
    args, _ = ap.parse_known_args()
    main(fast=args.fast, samples=args.samples, reuse=args.reuse)