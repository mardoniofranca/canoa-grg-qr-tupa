import numpy as np, json, time
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path, connected_components

def ba_edges(n, m, seed=None, track=None, times=None):
    """Barabasi-Albert por 'lista de nos repetidos'. Semente: estrela com m+1 nos (como networkx).
    Cada novo no escolhe m alvos DISTINTOS com prob. proporcional ao grau. Devolve (arestas, graus)."""
    rng = np.random.default_rng(seed)
    src, dst = [], []
    rep = []                                    # cada no aparece grau(no) vezes
    for v in range(1, m + 1):                   # estrela: no 0 no centro
        src.append(0); dst.append(v); rep += [0, v]
    graus = np.zeros(n, dtype=np.int64)
    graus[0] = m; graus[1:m + 1] = 1
    rec = {t: [] for t in (track or [])}
    tp = 0
    for v in range(m + 1, n):
        alvos = set()
        while len(alvos) < m:
            alvos.add(rep[rng.integers(len(rep))])
        for a in alvos:
            src.append(v); dst.append(a); rep += [v, a]
            graus[a] += 1
        graus[v] = m
        if track is not None and times is not None and tp < len(times) and v + 1 == times[tp]:
            for t in track: rec[t].append((v + 1, int(graus[t])))
            tp += 1
    return np.array(src), np.array(dst), graus, rec

def adj(n, s, d):
    A = sp.coo_matrix((np.ones(len(s)), (s, d)), shape=(n, n)).tocsr()
    return (A + A.T).tocsr()

def clustering(A):
    deg = np.asarray(A.sum(axis=1)).ravel()
    tri = np.asarray((A @ A).multiply(A).sum(axis=1)).ravel()      # (A^3)_ii = 2 x triangulos
    den = deg * (deg - 1)
    Ci = np.divide(tri, den, out=np.zeros_like(tri), where=den > 0)
    return Ci.mean()

def avg_path(A, k=100, seed=0):
    n = A.shape[0]
    idx = np.random.default_rng(seed).choice(n, size=min(k, n), replace=False)
    D = shortest_path(A, method="D", unweighted=True, directed=False, indices=idx)
    return D[np.isfinite(D) & (D > 0)].mean()

def er_graph(n, kmed, seed):
    rng = np.random.default_rng(seed)
    p = kmed / (n - 1)
    ne = rng.binomial(n * (n - 1) // 2, p)
    s = rng.integers(0, n, ne); d = rng.integers(0, n, ne)
    ok = s != d
    return adj(n, s[ok], d[ok])

if __name__ == "__main__":
    t0 = time.time(); out = {}
    # 1) distribuicao de grau, N=1e5, m=3, 5 realizacoes acumuladas
    m, N = 3, 100000
    allg = []
    for s in range(5):
        _, _, g, _ = ba_edges(N, m, seed=100 + s); allg.append(g)
    allg = np.concatenate(allg)
    np.save("graus_1e5.npy", allg)
    print("dist ok", round(time.time() - t0, 1), "s")
    # estimador MLE discreto (Clauset et al. 2009, eq. 3.7) e OLS ingenuo
    for kmin in (6, 10, 20):
        x = allg[allg >= kmin]
        a = 1 + len(x) / np.sum(np.log(x / (kmin - 0.5)))
        out[f"mle_kmin{kmin}"] = a
    print({k: round(v, 3) for k, v in out.items()})
    # 2) crescimento k_i(t)
    times = np.unique(np.round(np.logspace(np.log10(20), 5, 40)).astype(int))
    _, _, _, rec = ba_edges(100000, 3, seed=7, track=[10, 100, 1000], times=list(times))
    json.dump({str(k): v for k, v in rec.items()}, open("crescimento.json", "w"))
    print("crescimento ok", round(time.time() - t0, 1), "s")
    # 3) tabela vs N
    tab = []
    for n in (1000, 10000, 100000):
        kmax, C, L, Ler, Cer, kmean, Lc = [], [], [], [], [], [], []
        for s in range(5 if n < 100000 else 3):
            sr, ds, g, _ = ba_edges(n, 3, seed=200 + s)
            A = adj(n, sr, ds)
            kmax.append(g.max()); kmean.append(g.mean()); C.append(clustering(A)); L.append(avg_path(A, 100, s))
            E = er_graph(n, g.mean(), 300 + s)
            ncomp, lab = connected_components(E)
            big = np.argmax(np.bincount(lab)); keep = np.flatnonzero(lab == big)
            Eg = E[keep][:, keep]
            Cer.append(clustering(E)); Ler.append(avg_path(Eg, 100, s))
        lnn = np.log(n)
        tab.append(dict(N=n, kmean=float(np.mean(kmean)), kmax=float(np.mean(kmax)), kmax_teo=3 * np.sqrt(n),
                        C_ba=float(np.mean(C)), C_er=float(np.mean(Cer)), L_ba=float(np.mean(L)), L_er=float(np.mean(Ler)),
                        lnN=float(lnn), lnN_lnlnN=float(lnn / np.log(lnn))))
        print(tab[-1], round(time.time() - t0, 1), "s")
    out["tabela"] = tab
    # 4) kmax vs N
    Ns = [100, 300, 1000, 3000, 10000, 30000, 100000]
    km = []
    for n in Ns:
        v = [ba_edges(n, 3, seed=400 + s)[2].max() for s in range(8 if n <= 10000 else 4)]
        km.append((n, float(np.mean(v)), float(np.std(v))))
    out["kmax_vs_N"] = km
    json.dump(out, open("resultados.json", "w"), indent=1)
    print("fim", round(time.time() - t0, 1), "s")
