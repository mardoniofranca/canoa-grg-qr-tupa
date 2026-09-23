"""
Reprodução dos modelos e das figuras de:

  H. Jnane, G. Di Molfetta & F. M. Miatto (2020)
  "Growing Random Graphs with Quantum Rules", EPTCS 315, pp. 38-47.

Modelo (seções 2.1 e 2.2 do artigo):
  1. Hamiltoniano = matriz de adjacência A do grafo; U(t) = exp(-i A t).
  2. O(s) caminhante(s) evolui(em) por um tempo t ~ Exp(média tau).
  3. Mede-se a posição de cada caminhante: P(v) = |<v|U(t)|psi>|^2.
  4. Liga-se UM novo nó a todos os nós medidos (1 caminhante -> árvores;
     2 ou mais -> grafos com ciclos; se colidirem, o novo nó tem 1 aresta).
  5. Cada caminhante reinicia no nó onde colapsou.
  O grafo inicial é um único nó.

Figuras geradas (pasta SAIDA):
  fig2_arvores_1_caminhante.png     Fig. 2  (n=100, tau = 0.001 ... 10)
  fig3_estrelas_espectro.png        Fig. 3  (estrelas conectadas + autovalores)
  fig4_grafos_2_caminhantes.png     Fig. 4  (n=100, tau = 0.001 ... 10)
  fig5_distribuicao_grau.png        Fig. 5  (distribuição de grau, 1/2/3 caminhantes)
  fig6_diametro.png                 Fig. 6  (diâmetro x tau)
  fig7_fracao_folhas.png            Fig. 7  (fração de folhas x tau, 1 caminhante)
  fig8_clustering.png               Fig. 8  (clustering x tau)
  extra_tamanho_estrela.png         Checagem de E[n] ~ 1/tau (seção 2.1)

Uso (terminal):
  python reproduz_artigo_jnane.py              # configuração padrão
  python reproduz_artigo_jnane.py --rapido     # versão leve (poucos minutos)
  python reproduz_artigo_jnane.py --amostras 20
  python reproduz_artigo_jnane.py --reusar     # só refaz os gráficos (usa resultados.pkl)

Uso (Jupyter): cole/execute o arquivo numa célula e chame, por exemplo,
  main(rapido=True)   ou   main()   ou   main(amostras=20, reusar=False)

Dependências: numpy, scipy, networkx, matplotlib
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

SAIDA = "artigo"
TAUS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 3.0, 10.0]
TAUS_GRAU = [0.1, 0.5, 1.0, 10.0]                    # Fig. 5
TAUS_AMOSTRA = [0.001, 0.01, 0.05, 0.1, 0.5, 10.0]   # Figs. 2 e 4
CAMINHANTES = [1, 2, 3]


# =====================================================================
#  Modelo: crescimento por caminhada quântica de tempo contínuo
# =====================================================================
def _distribuicao(A, n, arestas, posicoes, t, grau_max):
    """P(v) de cada caminhante após evoluir por t (matriz n x n_caminhantes).
    Escolhe automaticamente o método mais barato; ambos são exatos."""
    w = len(posicoes)
    custo_expm = 4e-5 * t * grau_max + 5e-4           # estimativa empírica (s)
    custo_eigh = 3.5e-2 * (n / 600) ** 3

    if custo_expm < custo_eigh:                       # caminho esparso
        lin, col = zip(*arestas)
        M = sp.csr_matrix((np.ones(len(lin)), (lin, col)), shape=(n, n))
        M = M + M.T
        psi0 = np.zeros((n, w), dtype=complex)
        psi0[posicoes, np.arange(w)] = 1.0
        psi = expm_multiply(-1j * t * M, psi0)
        lam = None
    else:                                             # diagonalização densa
        lam, V = np.linalg.eigh(A[:n, :n])
        psi = V @ (np.exp(-1j * lam * t)[:, None] * V[posicoes, :].T)

    p = np.abs(psi) ** 2
    return p / p.sum(axis=0, keepdims=True)


def metricas(A, n):
    """Métricas de um grafo com n nós (matriz de adjacência densa)."""
    B = A[:n, :n]
    graus = B.sum(axis=1).astype(int)
    tri = np.einsum("ij,ji->i", B @ B, B)             # (A^3)_ii = 2 x triângulos
    denom = graus * (graus - 1)
    Ci = np.divide(tri, denom, out=np.zeros(n), where=denom > 0)
    dist = shortest_path(sp.csr_matrix(B), unweighted=True, directed=False)
    return dict(graus=graus,
                folhas=float(np.mean(graus == 1)),
                clustering=float(Ci.mean()),
                diametro=float(dist.max()))


def crescer(n_final, tau, n_caminhantes=1, seed=None, checkpoints=(),
            registrar_espectro=False):
    rng = np.random.default_rng(seed)
    A = np.zeros((n_final, n_final))
    graus = np.zeros(n_final, dtype=int)
    arestas = []
    n = 1
    posicoes = [0] * n_caminhantes
    seq, snaps, espectro = [], {}, []

    while n < n_final:
        t = rng.exponential(tau)
        if n == 1:
            novos = [0] * n_caminhantes
        else:
            P = _distribuicao(A, n, arestas, posicoes, t, max(1, graus[:n].max()))
            novos = [int(rng.choice(n, p=P[:, k])) for k in range(n_caminhantes)]
        for v in set(novos):
            A[n, v] = A[v, n] = 1.0
            graus[v] += 1
            graus[n] += 1
            arestas.append((v, n))
        seq.append(novos[0])
        posicoes = novos
        n += 1

        if registrar_espectro:
            lam = np.clip(np.linalg.eigvalsh(A[:n, :n])[::-1][:3], 0, None)
            espectro.append(np.pad(lam, (0, 3 - len(lam))))
        if n in checkpoints:
            snaps[n] = metricas(A, n)

    return dict(A=A, snaps=snaps, seq=seq, espectro=np.array(espectro))


# =====================================================================
#  Lote de simulações (Figs. 5 a 8)
# =====================================================================
def rodar_lote(n_final, checkpoints, amostras):
    resultados = {}          # (caminhantes, tau) -> lista de dict(snaps, seq)
    total = len(CAMINHANTES) * len(TAUS) * amostras
    feito, t0 = 0, time.time()
    for w in CAMINHANTES:
        for tau in TAUS:
            lista = []
            for s in range(amostras):
                r = crescer(n_final, tau, w, seed=[w, TAUS.index(tau), s],
                            checkpoints=checkpoints)
                lista.append(dict(snaps=r["snaps"], seq=r["seq"]))
                feito += 1
            resultados[(w, tau)] = lista
            dt = time.time() - t0
            print(f"  [{feito:>4}/{total}] caminhantes={w} tau={tau:<6g} "
                  f"({dt:5.0f}s decorridos)", flush=True)
    return resultados


def media_std(lista, n, chave):
    v = np.array([r["snaps"][n][chave] for r in lista], dtype=float)
    return v.mean(), v.std()


# =====================================================================
#  Figuras
# =====================================================================
def _layout(G, seed=1):
    return nx.spring_layout(G, seed=seed, iterations=150)


def fig_amostras(n_caminhantes, arquivo, titulo, n_nos=101):
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    for ax, tau in zip(axs.ravel(), TAUS_AMOSTRA):
        A = crescer(n_nos, tau, n_caminhantes, seed=7)["A"]
        G = nx.from_numpy_array(A)
        nx.draw(G, _layout(G), ax=ax, node_size=6, width=0.5, node_color="#1f77b4")
        ax.set_title(f"τ = {tau:g}", fontsize=10)
    fig.suptitle(titulo, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(SAIDA, arquivo), dpi=130)
    plt.close(fig)


def fig_estrelas_espectro(arquivo, tau=0.01, n_nos=300, tentativas=40):
    """Procura uma realização com ~3 estrelas (como na Fig. 3) e mostra
    o grafo e a evolução dos 3 maiores autovalores."""
    melhor, melhor_score = None, -1
    for seed in range(tentativas):
        r = crescer(n_nos, tau, 1, seed=seed, registrar_espectro=True)
        fim = np.sort(r["espectro"][-1])
        score = min(fim[-3:])                 # 3º autovalor grande = 3 estrelas
        if score > melhor_score:
            melhor, melhor_score = r, score
    r = melhor
    G = nx.from_numpy_array(r["A"])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    nx.draw(G, _layout(G), ax=a1, node_size=5, width=0.4, node_color="#1f77b4")
    a1.set_title(f"Grafo após {n_nos} nós (τ = {tau:g})")
    for i, cor in enumerate(["C0", "C1", "C2"]):
        a2.plot(np.arange(2, n_nos + 1), r["espectro"][:, i], color=cor,
                label=f"autovalor {i + 1}")
    a2.set_xlabel("índice (nº de nós)")
    a2.set_ylabel("valor")
    a2.set_title("Evolução dos 3 maiores autovalores")
    a2.legend()
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(SAIDA, arquivo), dpi=130)
    plt.close(fig)


def fig_grau(res, n, arquivo):
    fig, axs = plt.subplots(3, 1, figsize=(6.5, 13))
    cores = plt.cm.viridis(np.linspace(0.85, 0.05, len(TAUS_GRAU)))
    kmax = 60
    for ax, w in zip(axs, CAMINHANTES):
        for tau, cor in zip(TAUS_GRAU, cores):
            H = np.array([np.bincount(r["snaps"][n]["graus"], minlength=kmax + 1)[:kmax + 1] / n
                          for r in res[(w, tau)]])
            m, s = H.mean(0), H.std(0)
            k = np.arange(kmax + 1)
            ok = (k >= 1) & (m > 0)
            ax.plot(k[ok], m[ok], color=cor, label=f"{tau:g}")
            ax.fill_between(k[ok], np.clip(m - s, 1e-4, None)[ok], (m + s)[ok],
                            color=cor, alpha=0.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1)
        ax.set_xlabel("grau")
        ax.set_ylabel("prob. do grau")
        ax.set_title(f"Distribuição de grau — {w} caminhante(s) — {n} nós")
        ax.legend(title="τ")
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(SAIDA, arquivo), dpi=130)
    plt.close(fig)


def _curva_vs_tau(res, n, chave, arquivo, titulo, ylabel, logy=False, walkers=CAMINHANTES):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for w in walkers:
        M = np.array([media_std(res[(w, tau)], n, chave) for tau in TAUS])
        ax.plot(TAUS, M[:, 0], marker="o", ms=3, label=f"{w} caminhante(s)")
        ax.fill_between(TAUS, M[:, 0] - M[:, 1], M[:, 0] + M[:, 1], alpha=0.2)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("τ")
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(SAIDA, arquivo), dpi=130)
    plt.close(fig)


def fig_folhas(res, tamanhos, arquivo):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cores = plt.cm.magma(np.linspace(0.75, 0.15, len(tamanhos)))
    for n, cor in zip(tamanhos, cores):
        M = np.array([media_std(res[(1, tau)], n, "folhas") for tau in TAUS]) * 100
        ax.plot(TAUS, M[:, 0], color=cor, marker="o", ms=3, label=f"{n} nós")
        ax.fill_between(TAUS, M[:, 0] - M[:, 1], M[:, 0] + M[:, 1], color=cor, alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("τ")
    ax.set_ylabel("fração de folhas (%)")
    ax.set_title("Fração de folhas — 1 caminhante")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(SAIDA, arquivo), dpi=130)
    plt.close(fig)


def fig_tamanho_estrela(arquivo, n_final, amostras):
    """Checa E[n] ~ 1/tau (seção 2.1). Tamanho da estrela = nº de colapsos
    consecutivos no mesmo nó (o caminhante só sai da estrela ao colapsar num
    nó externo). Simulações próprias e mais longas para ter muitas estrelas."""
    taus = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    medias = []
    for i, tau in enumerate(taus):
        tamanhos = []
        for k in range(amostras):
            seq = np.array(crescer(n_final, tau, 1, seed=[99, i, k])["seq"])
            corte = np.flatnonzero(np.diff(seq) != 0) + 1
            corridas = np.diff(np.concatenate([[0], corte, [len(seq)]]))[:-1]  # descarta a última (incompleta)
            tamanhos.extend(corridas)
        medias.append(np.mean(tamanhos))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(taus, medias, "o-", label="simulação (tamanho médio da estrela)")
    ax.loglog(taus, 1 / np.array(taus), "k--", label="1/τ (artigo)")
    ax.set_xlabel("τ")
    ax.set_ylabel("E[n]")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(SAIDA, arquivo), dpi=130)
    plt.close(fig)
    return taus, medias


# =====================================================================
def main(rapido=False, amostras=None, reusar=False):
    """Roda tudo. No Jupyter/IPython, chame diretamente, por exemplo:
         main(rapido=True)        # versão leve
         main(amostras=20)        # curvas mais suaves
         main(reusar=True)        # só refaz os gráficos
    """
    if rapido:
        n_final, checks, n_amostras = 300, (75, 150, 225, 300), amostras or 4
        n_grau = 300
    else:
        n_final, checks, n_amostras = 600, (150, 300, 450, 500, 600), amostras or 8
        n_grau = 500

    os.makedirs(SAIDA, exist_ok=True)
    arq_pkl = os.path.join(SAIDA, "resultados.pkl")

    print("Figs. 2, 3 e 4 (amostras de grafos)...")
    fig_amostras(1, "fig2_arvores_1_caminhante.png",
                 "1 caminhante — 100 passos — crescimento em árvores")
    fig_amostras(2, "fig4_grafos_2_caminhantes.png",
                 "2 caminhantes — 100 passos — grafos com ciclos")
    fig_estrelas_espectro("fig3_estrelas_espectro.png")

    if reusar and os.path.exists(arq_pkl):
        print("Carregando", arq_pkl)
        with open(arq_pkl, "rb") as f:
            res = pickle.load(f)
    else:
        print(f"Lote de simulações: {n_final} nós, {n_amostras} amostras por ponto...")
        res = rodar_lote(n_final, checks, n_amostras)
        with open(arq_pkl, "wb") as f:
            pickle.dump(res, f)

    print("Figs. 5 a 8...")
    fig_grau(res, n_grau, "fig5_distribuicao_grau.png")
    _curva_vs_tau(res, n_grau, "diametro", "fig6_diametro.png",
                  f"Diâmetro — {n_grau} nós", "diâmetro", logy=True)
    fig_folhas(res, [c for c in checks if c != 500], "fig7_fracao_folhas.png")
    _curva_vs_tau(res, n_grau, "clustering", "fig8_clustering.png",
                  f"Coeficiente de clustering — {n_grau} nós", "clustering")

    print("Checagem E[n] ~ 1/tau...")
    taus, medias = fig_tamanho_estrela("extra_tamanho_estrela.png",
                                       600 if rapido else 1500, n_amostras)
    print("\nTamanho médio da estrela (simulação) x 1/tau:")
    for t, m in zip(taus, medias):
        print(f"  tau={t:<6g} E[n]={m:7.1f}   1/tau={1 / t:7.1f}   razão={m * t:.2f}")
    print(f"\nFiguras em ./{SAIDA}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="menos nós e amostras")
    ap.add_argument("--amostras", type=int, default=None)
    ap.add_argument("--reusar", action="store_true", help="reusa resultados.pkl")
    # parse_known_args ignora argumentos extras (ex.: '-f kernel.json' do Jupyter)
    args, _ = ap.parse_known_args()
    main(rapido=args.rapido, amostras=args.amostras, reusar=args.reusar)
