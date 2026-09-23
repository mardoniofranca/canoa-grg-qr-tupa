import json, numpy as np, networkx as nx, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import poisson
from sim import ba_edges, adj, er_graph

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
AZ, LA, VM, CZ = "#1F3864", "#2F5496", "#C0392B", "#7F7F7F"
res = json.load(open("resultados.json"))

# ---------------- F1: esquema de um passo -----------------
def fig_esquema():
    E = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (3, 5)]
    pos = {0: (0, 0), 1: (-1.2, 0.8), 2: (-1.2, -0.8), 3: (1.2, 0.5), 4: (0.3, -1.3), 5: (2.3, 1.0)}
    G = nx.Graph(E); deg = dict(G.degree()); tot = sum(deg.values())
    rng = np.random.default_rng(11)
    nodes = list(G.nodes()); p = np.array([deg[v] for v in nodes]) / tot
    alvos = [int(a) for a in rng.choice(nodes, size=2, replace=False, p=p)]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, depois in zip(axs, (False, True)):
        H = G.copy(); pp = dict(pos)
        if depois:
            H.add_node(6); H.add_edges_from((6, a) for a in alvos); pp[6] = (2.0, -0.9)
        ne = [e for e in H.edges() if 6 not in e]
        nx.draw_networkx_edges(H, pp, edgelist=ne, ax=ax, width=1.6, edge_color=CZ)
        if depois:
            nx.draw_networkx_edges(H, pp, edgelist=[(6, a) for a in alvos], ax=ax, width=3, edge_color=VM)
        nx.draw_networkx_nodes(H, pp, nodelist=nodes, ax=ax, node_size=[300 + 260 * deg[v] for v in nodes], node_color="#9DC3E6", edgecolors=AZ, linewidths=1.2)
        if depois: nx.draw_networkx_nodes(H, pp, nodelist=[6], ax=ax, node_size=420, node_color=VM, edgecolors="k")
        for v in nodes:
            x, y = pp[v]
            ax.text(x, y, str(v), ha="center", va="center", fontsize=11, fontweight="bold", color=AZ)
            if not depois:
                ax.text(x, y - 0.42 - 0.03 * deg[v], f"k={deg[v]}, Π={100 * deg[v] / tot:.0f}%", ha="center", fontsize=8.5, color="#333")
        if depois: ax.text(2.0, -0.9, "6", ha="center", va="center", color="w", fontweight="bold")
        ax.set_xlim(-2.2, 3.0); ax.set_ylim(-2.1, 1.8); ax.axis("off")
        ax.set_title("(a) Antes: Π = k / Σk (Σk = 12)" if not depois else f"(b) Depois: nó 6 liga-se a {alvos} (m = 2)", fontsize=10.5)
    fig.tight_layout(); fig.savefig("fig_esquema.png", dpi=150); plt.close(fig)
    return alvos

# ---------------- F2: BA x ER -----------------
def fig_redes():
    n = 120
    s, d, g, _ = ba_edges(n, 2, seed=4)
    GB = nx.Graph(); GB.add_edges_from(zip(s.tolist(), d.tolist()))
    GE = nx.gnp_random_graph(n, g.mean() / (n - 1), seed=3)
    GE = GE.subgraph(max(nx.connected_components(GE), key=len)).copy()
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.4))
    for ax, G, tit in ((axs[0], GB, "Barabási–Albert (m = 2)"), (axs[1], GE, "Erdős–Rényi (mesmo ⟨k⟩ ≈ 4; componente gigante)")):
        deg = dict(G.degree()); pos = nx.spring_layout(G, seed=2, k=0.35, iterations=200)
        nx.draw_networkx_edges(G, pos, ax=ax, width=0.5, edge_color="#999", alpha=0.7)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[14 + 22 * deg[v] for v in G], node_color=[deg[v] for v in G], cmap="viridis_r", vmin=1, vmax=max(dict(GB.degree()).values()), edgecolors="k", linewidths=0.3)
        top = sorted(deg, key=deg.get, reverse=True)[:3]
        ax.set_title(f"{tit}\ngrau máximo = {max(deg.values())}", fontsize=10.5); ax.axis("off")
    fig.tight_layout(); fig.savefig("fig_redes.png", dpi=150); plt.close(fig)
    return int(max(dict(GB.degree()).values())), int(max(dict(GE.degree()).values()))

# ---------------- F3: distribuicao de grau -----------------
def fig_grau():
    g = np.load("graus_1e5.npy"); m = 3; tot = len(g)
    ks = np.arange(m, 31); emp = np.array([(g == k).sum() for k in ks]) / tot
    kk = np.arange(m, 3000)
    exato = 2 * m * (m + 1) / (kk * (kk + 1) * (kk + 2)); mf = 2 * m**2 / kk.astype(float)**3
    edges = np.unique(np.round(np.logspace(np.log10(31), np.log10(g.max() + 1), 22)).astype(int))
    xb, yb = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        c = ((g >= a) & (g < b)).sum()
        if c > 0: xb.append(np.sqrt(a * (b - 1 if b - 1 > a else a))); yb.append(c / tot / (b - a))
    fig, ax = plt.subplots(figsize=(7.2, 5))
    ax.loglog(ks, emp, "o", color=AZ, ms=5, label="simulação (k ≤ 30)")
    ax.loglog(xb, yb, "s", color=LA, ms=5, mfc="none", label="simulação (k > 30, bins logarítmicos)")
    ax.loglog(kk, exato, "-", color=VM, lw=1.6, label=r"teoria exata: $2m(m+1)/[k(k+1)(k+2)]$")
    ax.loglog(kk, mf, "--", color="#E69F00", lw=1.5, label=r"campo médio: $2m^2 k^{-3}$")
    kp = np.arange(0, 40)
    ax.loglog(kp[kp > 0], poisson.pmf(kp[kp > 0], 6), "^-", color="#2E8B57", ms=4, lw=1, label="Poisson (grafo aleatório, ⟨k⟩ = 6)")
    ax.set_xlim(2, 3000); ax.set_ylim(1e-11, 1); ax.set_xlabel("grau k"); ax.set_ylabel("P(k)")
    ax.set_title("Distribuição de grau — N = 100 000, m = 3 (5 realizações)"); ax.legend(fontsize=8.2, loc="lower left")
    fig.tight_layout(); fig.savefig("fig_grau.png", dpi=150); plt.close(fig)

# ---------------- F4: crescimento k_i(t) -----------------
def fig_cresc():
    rec = json.load(open("crescimento.json")); m = 3
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    cores = {"10": VM, "100": "#E69F00", "1000": "#2E8B57"}
    for ti, cor in cores.items():
        t = np.array(rec[ti]["t"]); k = np.array(rec[ti]["mean"]); t0 = int(ti); ok = t > t0
        ax.loglog(t[ok], k[ok], "o", color=cor, ms=4, label=f"média de 300 simulações, nó nascido em t = {ti}")
        tt = np.logspace(np.log10(t0), np.log10(t.max()), 100); ax.loglog(tt, m * np.sqrt(tt / t0), "-", color=cor, lw=1.2)
    ax.set_xlabel("tempo t (nº de nós)"); ax.set_ylabel("grau médio k(t)")
    ax.set_title(r"Crescimento do grau: simulação vs $k_i(t)=m\,(t/t_i)^{1/2}$ (linhas)")
    ax.legend(fontsize=8.5); fig.tight_layout(); fig.savefig("fig_crescimento.png", dpi=150); plt.close(fig)

# ---------------- F5: robustez -----------------
def gcc_curve(n, edges, order):
    parent = list(range(n)); size = [1] * n; present = [False] * n
    nb = [[] for _ in range(n)]
    for a, b in edges: nb[a].append(b); nb[b].append(a)
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    best = 0; arr = [0]
    for v in reversed(order):
        present[v] = True
        for u in nb[v]:
            if present[u]:
                ra, rb = find(u), find(v)
                if ra != rb:
                    if size[ra] < size[rb]: ra, rb = rb, ra
                    parent[rb] = ra; size[ra] += size[rb]
        best = max(best, size[find(v)]); arr.append(best)
    return np.array(arr)          # arr[j] = GCC apos recolocar j nos

def robustez(n=5000, m=2, runs=5):
    fs = np.linspace(0, 0.95, 96); out = {k: [] for k in ("ba_r", "ba_a", "er_r", "er_a")}
    for r in range(runs):
        s, d, g, _ = ba_edges(n, m, seed=500 + r); Eb = list(zip(s.tolist(), d.tolist()))
        A = er_graph(n, g.mean(), 600 + r).tocoo(); Ee = [(int(a), int(b)) for a, b in zip(A.row, A.col) if a < b]
        for nome, E in (("ba", Eb), ("er", Ee)):
            deg = np.zeros(n)
            for a, b in E: deg[a] += 1; deg[b] += 1
            rng = np.random.default_rng(r)
            ord_r = rng.permutation(n).tolist()
            ord_a = sorted(range(n), key=lambda v: (-deg[v], rng.random()))
            for tag, order in (("r", ord_r), ("a", ord_a)):
                arr = gcc_curve(n, E, order)
                out[f"{nome}_{tag}"].append([arr[n - int(round(f * n))] / n for f in fs])
    return fs, {k: np.mean(v, axis=0) for k, v in out.items()}

def fig_rob():
    fs, S = robustez()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(fs, 1 - fs, ":", color="#555", label="limite superior 1 − f")
    ax.plot(fs, S["ba_r"], "-", color=AZ, lw=2, label="BA — falhas aleatórias")
    ax.plot(fs, S["ba_a"], "--", color=AZ, lw=2, label="BA — ataque a hubs")
    ax.plot(fs, S["er_r"], "-", color=VM, lw=1.6, label="ER — falhas aleatórias")
    ax.plot(fs, S["er_a"], "--", color=VM, lw=1.6, label="ER — ataque por grau")
    ax.set_xlabel("fração f de nós removidos"); ax.set_ylabel("maior componente / N original")
    ax.set_title("Robustez: N = 5000, ⟨k⟩ ≈ 4 (média de 5 realizações)"); ax.legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig("fig_robustez.png", dpi=150); plt.close(fig)
    tab = {}
    for f in (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90):
        i = int(round(f * 100)); tab[f] = {k: float(v[i]) for k, v in S.items()}
    def f_frag(v, lim=0.05): return float(fs[np.argmax(v < lim)]) if (v < lim).any() else None
    tab["frag"] = {k: f_frag(v) for k, v in S.items()}
    return tab

def fig_kmax():
    km = np.array(res["kmax_vs_N"]); m = 3
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    ax.errorbar(km[:, 0], km[:, 1], yerr=km[:, 2], fmt="o", color=AZ, capsize=3, label="grau máximo simulado")
    nn = np.logspace(2, 5, 50); ax.loglog(nn, m * np.sqrt(nn), "-", color=VM, label=r"$m\sqrt{N}$ (teoria)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("N"); ax.set_ylabel(r"$k_{max}$")
    ax.set_title("Tamanho do maior hub (m = 3)"); ax.legend(); fig.tight_layout(); fig.savefig("fig_kmax.png", dpi=150); plt.close(fig)

if __name__ == "__main__":
    info = {}
    info["esquema_alvos"] = fig_esquema()
    info["redes_kmax"] = fig_redes()
    fig_grau(); fig_cresc(); fig_kmax()
    info["robustez"] = fig_rob()
    json.dump(info, open("info.json", "w"), indent=1, default=str)
    print(json.dumps(info, indent=1, default=str))
