"""K-means exploratorio sobre as sinopses PT-BR (TF-IDF, sem dependencias novas).

Pergunta: as sinopses se organizam por TROPO/GENERO ou pelo ROTULO de conteudo sensivel?
Se os clusters seguirem tropos e forem ~independentes do rotulo, isso reforca o achado
"a sinopse nao avisa". Se seguirem o rotulo, a sinopse carrega mais sinal do que parece.

Uso:  .venv/bin/python scripts/03_kmeans_sinopses.py [--kmin 2 --kmax 10]
Saida: outputs/tables/kmeans_sinopses_{silhueta,clusters,crosstab}.csv
       outputs/figures/fig_21_kmeans_sinopses.{pdf,png}
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
from pipeline_ptbr.common import sem_acento  # noqa: E402
from pipeline_ptbr.eda_stats import STOPWORDS_EDITORIAL, STOP_TODAS  # noqa: E402

STOP = STOP_TODAS | STOPWORDS_EDITORIAL
TOKEN = re.compile(r"[a-z]{3,}")
RNG = np.random.default_rng(42)


# ----------------------------------------------------------------------------- texto
def tokens(txt: str) -> list[str]:
    return [t for t in TOKEN.findall(sem_acento(str(txt)).lower()) if t not in STOP]


def tfidf(docs: list[list[str]], min_df: int = 3, max_df: float = 0.5):
    df = Counter(t for d in docs for t in set(d))
    vocab = sorted(t for t, n in df.items() if n >= min_df and n <= max_df * len(docs))
    idx = {t: i for i, t in enumerate(vocab)}
    X = np.zeros((len(docs), len(vocab)))
    for i, d in enumerate(docs):
        for t, n in Counter(d).items():
            if t in idx:
                X[i, idx[t]] = 1 + np.log(n)  # tf sublinear
    idf = np.log(len(docs) / np.array([df[t] for t in vocab])) + 1
    X *= idf
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12  # L2 -> k-means ~ cosseno
    return X, np.array(vocab)


# ----------------------------------------------------------------------------- k-means
def kmeans(X: np.ndarray, k: int, n_init: int = 20):
    melhor = None
    for _ in range(n_init):
        cent, lab = kmeans2(X, k, minit="++", seed=RNG, iter=100)
        inercia = ((X - cent[lab]) ** 2).sum()
        if melhor is None or inercia < melhor[0]:
            melhor = (inercia, cent, lab)
    return melhor


def silhueta(X: np.ndarray, lab: np.ndarray) -> float:
    D = cdist(X, X, "cosine")
    s = []
    for i in range(len(X)):
        mesmo = lab == lab[i]
        mesmo[i] = False
        if not mesmo.any():
            s.append(0.0)
            continue
        a = D[i, mesmo].mean()
        b = min(D[i, lab == c].mean() for c in set(lab) if c != lab[i])
        s.append((b - a) / max(a, b))
    return float(np.mean(s))


def nmi(a: np.ndarray, b: np.ndarray) -> float:
    ct = pd.crosstab(a, b).to_numpy().astype(float)
    n = ct.sum()
    pa, pb = ct.sum(1) / n, ct.sum(0) / n
    p = ct / n
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(p * np.log(p / np.outer(pa, pb)))
    h = lambda q: -np.sum(q[q > 0] * np.log(q[q > 0]))
    return float(mi / np.sqrt(h(pa) * h(pb))) if h(pa) and h(pb) else 0.0


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--k", type=int, help="forca um k (senao usa o de maior silhueta)")
    args = ap.parse_args()

    d = pd.read_csv(RAIZ / "data/processed/livros_ptbr_v3_rotulado.csv")
    d = d.drop_duplicates("Hardcover_Book_ID").dropna(subset=["Sinopse_PT"]).reset_index(drop=True)
    docs = [tokens(s) for s in d.Sinopse_PT]
    X, vocab = tfidf(docs)
    print(f"{len(d)} obras com sinopse, vocabulario de {len(vocab)} termos")

    # escolha de k por silhueta (cosseno)
    sil = {}
    for k in range(args.kmin, args.kmax + 1):
        _, _, lab = kmeans(X, k)
        sil[k] = silhueta(X, lab)
        print(f"  k={k:2d}  silhueta={sil[k]:.3f}")
    k = args.k or max(sil, key=sil.get)
    _, cent, lab = kmeans(X, k)
    d["cluster"] = lab
    print(f"\nk escolhido = {k} (silhueta {sil[k]:.3f})")

    # termos mais caracteristicos por cluster (centroide TF-IDF)
    linhas = []
    for c in range(k):
        top = vocab[np.argsort(cent[c])[::-1][:10]]
        sub = d[d.cluster == c]
        linhas.append({
            "cluster": c, "n": len(sub),
            "termos": ", ".join(top),
            "pct_alto": round(100 * (sub.Sensivel_Label == "alto").mean(), 1),
            "pct_indeterminado": round(100 * (sub.Sensivel_Label == "indeterminado").mean(), 1),
            "pct_sexo": round(100 * sub.Flag_Eixo_Sexo.fillna(False).astype(bool).mean(), 1),
            "pct_violencia": round(100 * sub.Flag_Eixo_Violencia.fillna(False).astype(bool).mean(), 1),
            "pct_via_termo_conteudo": round(100 * sub.Via_Termo_Conteudo.fillna(False).astype(bool).mean(), 1),
            "mediana_leitores": sub.Qtd_Usuarios_Hardcover.median(),
        })
    clusters = pd.DataFrame(linhas).sort_values("n", ascending=False)
    print("\n" + clusters.to_string(index=False))

    # clusters x rotulo: NMI e tabela
    ct = pd.crosstab(d.cluster, d.Sensivel_Label)
    pop = pd.cut(d.Qtd_Usuarios_Hardcover.fillna(0), [-1, 9, 99, 999, np.inf], labels=["<10", "10-99", "100-999", ">=1000"])
    print(f"\nNMI(cluster, rotulo)       = {nmi(lab, d.Sensivel_Label.to_numpy()):.3f}")
    print(f"NMI(cluster, popularidade) = {nmi(lab, pop.astype(str).to_numpy()):.3f}")
    print(f"NMI(cluster, eixo sexo)    = {nmi(lab, d.Flag_Eixo_Sexo.fillna(False).astype(bool).to_numpy()):.3f}")
    print("(0 = independentes, 1 = identicos; para referencia, NMI(rotulo, popularidade) =",
          f"{nmi(d.Sensivel_Label.to_numpy(), pop.astype(str).to_numpy()):.3f})")

    tab = RAIZ / "outputs/tables"
    pd.Series(sil, name="silhueta").rename_axis("k").to_csv(tab / "kmeans_sinopses_silhueta.csv")
    clusters.to_csv(tab / "kmeans_sinopses_clusters.csv", index=False)
    ct.to_csv(tab / "kmeans_sinopses_crosstab.csv")
    d[["Hardcover_Book_ID", "Titulo_PT", "cluster", "Sensivel_Label"]].to_csv(tab / "kmeans_sinopses_obras.csv", index=False)

    # figura: SVD 2D, cor = rotulo, marcador = cluster
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Xc = X - X.mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    P = U[:, :2] * S[:2]
    cores = {"alto": "#D55E00", "indeterminado": "#999999", "provavel_baixo": "#0072B2"}
    marcas = "osD^v<>Pp*Xh"
    fig, ax = plt.subplots(figsize=(9, 7))
    for c in range(k):
        for rot, cor in cores.items():
            m = (d.cluster == c) & (d.Sensivel_Label == rot)
            ax.scatter(P[m, 0], P[m, 1], c=cor, marker=marcas[c % len(marcas)], s=55,
                       edgecolor="white", linewidth=0.5, alpha=0.9)
    for c in range(k):
        m = d.cluster == c
        cx, cy = P[m, 0].mean(), P[m, 1].mean()
        ax.annotate(f"C{c}: " + ", ".join(vocab[np.argsort(cent[c])[::-1][:3]]), (cx, cy),
                    fontsize=10, weight="bold", ha="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.4", alpha=0.85))
    for rot, cor in cores.items():
        ax.scatter([], [], c=cor, label=f"rótulo {rot}")
    for c in range(k):
        ax.scatter([], [], c="0.3", marker=marcas[c % len(marcas)], label=f"cluster C{c}")
    ax.legend(fontsize=8, ncol=2, loc="best")
    ax.set_title(f"K-means (k={k}) sobre TF-IDF das sinopses PT-BR — projeção SVD 2D\n"
                 f"silhueta={sil[k]:.2f} · NMI(cluster, rótulo)={nmi(lab, d.Sensivel_Label.to_numpy()):.2f}")
    ax.set_xlabel("componente 1"); ax.set_ylabel("componente 2")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(RAIZ / f"outputs/figures/fig_21_kmeans_sinopses.{ext}", dpi=300)
    print(f"\nfigura: outputs/figures/fig_21_kmeans_sinopses.pdf")


if __name__ == "__main__":
    main()
