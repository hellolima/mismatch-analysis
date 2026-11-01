"""Camada de apresentacao visual da EDA: figuras cientificas na paleta Okabe-Ito e WordClouds.

Salva todas as figuras em `outputs/figures/` nos formatos solicitados (padrao: PNG 300 dpi + PDF vetorial).
"""
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from scipy.stats import spearmanr

from . import config as C
from .dataset import deduplicar_por_obra, resolver_colunas_rotulo
from .eda_stats import SEED, cobertura_por_popularidade, prevalencias_flags

log = logging.getLogger("pipeline.eda.plots")

# Paleta Okabe-Ito: distinguivel em daltonismo e em impressao P&B razoavel
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#F0E442", "#000000"]
COR_PRIM = OKABE_ITO[0]
COR_SEC = OKABE_ITO[1]
COR_TERC = OKABE_ITO[2]
COR_CINZA = "#6E6E6E"
CMAP_SEQ = "Blues"
CMAP_WC_UNI = ["#003F66", "#0072B2", "#2E86C1", "#56B4E9"]
CMAP_WC_BI = ["#7A2E00", "#D55E00", "#E27D2E", "#E69F00"]

# Dimensoes ampliadas para leitura confortavel dos rotulos numericos
W1, W2 = 5.2, 8.6

RC_PUB = {
    "figure.dpi": 140, "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "axes.unicode_minus": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "axes.prop_cycle": matplotlib.cycler(color=OKABE_ITO),
    "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.constrained_layout.use": True,
}


def aplicar_estilo() -> None:
    sns.set_theme(style="ticks", context="paper", rc=RC_PUB)
    plt.rcParams.update(RC_PUB)


def salvar_fig(fig: plt.Figure, nome: str, formatos: Tuple[str, ...] = ("png", "pdf"),
               out_dir: Path = Path(C.OUT_FIGURES)) -> None:
    """Salva a figura em todos os formatos solicitados em outputs/figures/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formatos:
        fig.savefig(out_dir / f"{nome}.{fmt}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  Salvo: {out_dir / nome}.[{'/'.join(formatos)}]")


def titulo_ax(ax: plt.Axes, txt: str, exibir: bool = False) -> None:
    if exibir:
        ax.set_title(txt, fontweight="bold")


def barh_top(ax: plt.Axes, freq: Dict[str, int], n: int, cor: str, xlabel: str,
             n_base: int = 0) -> None:
    """Plota barras horizontais das top-N categorias com anotacao direta de contagem (e % se n_base > 0)."""
    itens = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:n][::-1]
    if not itens:
        return
    labels = [k for k, _ in itens]
    vals = [v for _, v in itens]
    ax.barh(labels, vals, color=cor, edgecolor="black", linewidth=0.5)
    vmax = max(vals) if vals else 1
    for y, v in enumerate(vals):
        if n_base > 0:
            txt = f" {v} ({100 * v / n_base:.1f}%)".replace(".", ",")
        else:
            txt = f" {v}"
        ax.text(v, y, txt, va="center", ha="left", fontsize=8.5, fontweight="medium")
    ax.set_xlim(0, vmax * (1.28 if n_base > 0 else 1.18))
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))


def _anotar_barras_verticais(ax: plt.Axes, xs, vals, total: int = 0, mostrar_pct: bool = False,
                             fontsize: float = 8.5) -> None:
    """Anota contagem (e opcionalmente percentual) no topo de barras verticais."""
    vmax = max(vals) if len(vals) else 1
    for x, v in zip(xs, vals):
        if v <= 0:
            continue
        if mostrar_pct and total > 0:
            txt = f"{int(v)}\n({100 * v / total:.1f}%)".replace(".", ",")
        else:
            txt = f"{int(v)}"
        ax.text(x, v + vmax * 0.02, txt, ha="center", va="bottom", fontsize=fontsize, fontweight="medium")
    ax.set_ylim(0, vmax * (1.28 if mostrar_pct else 1.18))


def _cor_wc(paleta: List[str]):
    def fn(word=None, font_size=None, position=None, orientation=None, random_state=None, **kw):
        rng = np.random.default_rng(abs(hash(word)) % (2 ** 32))
        return paleta[int(rng.integers(0, len(paleta)))]
    return fn


def _wc_img(freq: Dict[str, int], paleta: List[str], max_words: int, w: int = 1200, h: int = 800):
    if not freq:
        return None
    return WordCloud(width=w, height=h, background_color="white", color_func=_cor_wc(paleta),
                     max_words=max_words, prefer_horizontal=0.9, relative_scaling=0.6,
                     min_font_size=8, random_state=SEED, collocations=False,
                     font_path=None).generate_from_frequencies(dict(freq))


def plotar_wordclouds(para_plot: Dict[str, Tuple[Dict[str, int], Dict[str, int], int]],
                      formatos: Tuple[str, ...] = ("png", "pdf"),
                      fig_titulos: bool = False,
                      out_dir: Path = Path(C.OUT_FIGURES)) -> None:
    log.info("\n[ANALISE 4] Gerando figuras de Wordclouds (unigramas + bigramas)...")
    for nome, (uni_f, bi_f, max_words) in para_plot.items():
        fig, axes = plt.subplots(1, 2, figsize=(W2, 3.8))
        for ax, (freq, paleta, letra, rot) in zip(
                axes, [(uni_f, CMAP_WC_UNI, "a", "Palavras (unigramas)"),
                       (bi_f, CMAP_WC_BI, "b", "Bigramas")]):
            wc = _wc_img(freq, paleta, max_words)
            ax.axis("off")
            if wc is not None:
                ax.imshow(wc, interpolation="bilinear")
            else:
                ax.text(0.5, 0.5, "sem termos acima do limiar", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"({letra}) {rot}", fontsize=10, loc="left")
        if fig_titulos:
            fig.suptitle(f"Nuvens de termos: {nome} (frequência de documento, nível obra)", fontweight="bold")
        salvar_fig(fig, f"fig_wordcloud_{nome}", formatos, out_dir)


def gerar_visualizacoes(df: pd.DataFrame, stats_tags: Dict, stats_dist: Dict, stats_cat: Dict,
                        extra: Dict, formatos: Tuple[str, ...] = ("png", "pdf"),
                        fig_titulos: bool = False,
                        out_dir: Path = Path(C.OUT_FIGURES)) -> None:
    log.info("\n[ANALISE 5] Gerando Visualizacoes...")
    ob = deduplicar_por_obra(df)
    n_obras = len(ob)
    n_edicoes = len(df)

    # FIG 1: Ratings (histograma com numeros nas barras + boxplot horizontal compartilhando o eixo)
    if "Rating_Normalizado" in ob.columns:
        r = ob["Rating_Normalizado"].dropna()
        if len(r):
            fig, (ax, axb) = plt.subplots(2, 1, figsize=(W1 + 0.8, 3.8), sharex=True,
                                          gridspec_kw={"height_ratios": [4, 1]})
            counts, edges, _ = ax.hist(r, bins=np.arange(1.0, 5.25, 0.25), color=COR_PRIM,
                                       edgecolor="black", linewidth=0.5, alpha=0.9)
            centers = 0.5 * (edges[:-1] + edges[1:])
            _anotar_barras_verticais(ax, centers, counts, fontsize=8)
            ax.axvline(r.mean(), color=COR_SEC, ls="--", lw=1.4, label=f"Média = {r.mean():.2f} (n={len(r)})")
            ax.axvline(r.median(), color=COR_TERC, ls=":", lw=1.6, label=f"Mediana = {r.median():.2f} (DP={r.std():.2f})")
            ax.set_ylabel("Nº de obras")
            ax.legend(loc="upper left")
            titulo_ax(ax, "Distribuição de ratings (nível obra)", fig_titulos)
            axb.boxplot(r, vert=False, widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor=COR_PRIM, alpha=0.5), medianprops=dict(color="black"))
            axb.set_yticks([])
            axb.set_xlabel("Rating (Hardcover, 1–5)")
            axb.grid(axis="y", visible=False)
            salvar_fig(fig, "fig_01_ratings_dist", formatos, out_dir)

    # FIG 2: Distribuicao temporal (edicoes PT-BR por ano) com contagem e % no topo
    if "Ano" in df.columns and df["Ano"].notna().any():
        a = df["Ano"].dropna().astype(int)
        idx = range(int(a.min()), int(a.max()) + 1)
        cont = a.value_counts().reindex(idx, fill_value=0)
        fig, ax = plt.subplots(figsize=(W2, 3.4))
        xs = np.arange(len(cont))
        ax.bar(xs, cont.values, color=COR_PRIM, edgecolor="black", linewidth=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(cont.index.astype(str), rotation=35, ha="right")
        _anotar_barras_verticais(ax, xs, cont.values, total=len(a), mostrar_pct=True, fontsize=8)
        ax.set_xlabel("Ano de publicação da edição PT-BR")
        ax.set_ylabel("Nº de edições")
        ax.grid(axis="x", visible=False)
        titulo_ax(ax, "Distribuição temporal das edições PT-BR", fig_titulos)
        salvar_fig(fig, "fig_02_anos_dist", formatos, out_dir)

    # FIG 3: Top 10 editoras com contagem e %
    if stats_cat.get("editora"):
        fig, ax = plt.subplots(figsize=(W1 + 1.0, 3.6))
        barh_top(ax, stats_cat["editora"], 10, COR_PRIM, "Nº de edições (% do total)", n_base=n_edicoes)
        titulo_ax(ax, "Top 10 editoras (grafias agrupadas)", fig_titulos)
        salvar_fig(fig, "fig_03_editoras_top10", formatos, out_dir)

    # FIG 4: Top tags com contagem e % de obras
    cats = [c for c in ("Generos", "Moods", "Tags_Gerais") if c in stats_tags]
    if cats:
        fig, axes = plt.subplots(len(cats), 1, figsize=(W1 + 1.4, 3.2 * len(cats)))
        axes = np.atleast_1d(axes)
        nomes = {"Generos": "Gêneros", "Moods": "Moods", "Tags_Gerais": "Tags gerais"}
        for i, (ax, c) in enumerate(zip(axes, cats)):
            barh_top(ax, stats_tags[c]["frequencias"], 12, OKABE_ITO[i],
                     f"Nº de obras (% de {n_obras} obras)", n_base=n_obras)
            ax.set_title(f"({'abc'[i]}) {nomes[c]} (cobertura: {stats_tags[c]['n_registros']}/{n_obras} = {stats_tags[c]['cobertura_%']:.1f}%)",
                         loc="left", fontsize=10, fontweight="bold")
        salvar_fig(fig, "fig_04_tags_top", formatos, out_dir)

    # FIG 5: Rating vs nº de avaliacoes
    if {"Rating_Normalizado", "Qtd_Avaliacoes_Clean"} <= set(ob.columns):
        d = ob.dropna(subset=["Rating_Normalizado", "Qtd_Avaliacoes_Clean"])
        d = d[d["Qtd_Avaliacoes_Clean"] > 0]
        if len(d) > 2:
            fig, ax = plt.subplots(figsize=(W1 + 0.8, 3.4))
            ax.scatter(d["Qtd_Avaliacoes_Clean"], d["Rating_Normalizado"], s=24, alpha=0.75,
                       color=COR_PRIM, edgecolors="black", linewidths=0.4)
            ax.set_xscale("log")
            rho, p = spearmanr(d["Qtd_Avaliacoes_Clean"], d["Rating_Normalizado"])
            ptxt = "p < 0,001" if p < 0.001 else f"p = {p:.3f}".replace(".", ",")
            ax.text(0.04, 0.06, f"Spearman ρ = {rho:.2f}".replace(".", ",") + f" ({ptxt}; n = {len(d)})",
                    transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle="round", fc="white", ec=COR_CINZA, lw=0.6))
            ax.set_xlabel("Nº de avaliações (escala log)")
            ax.set_ylabel("Rating (1–5)")
            titulo_ax(ax, "Rating × nº de avaliações", fig_titulos)
            salvar_fig(fig, "fig_05_rating_vs_qtd_scatter", formatos, out_dir)

    # FIG 6: Content warnings com contagem e % de obras
    cw = stats_tags.get("Content_Warnings", {}).get("frequencias")
    if cw:
        fig, ax = plt.subplots(figsize=(W1 + 1.4, 4.4))
        barh_top(ax, cw, 15, COR_SEC, f"Nº de obras (% de {n_obras} obras)", n_base=n_obras)
        ax.set_title(f"Top 15 Content Warnings (presentes em {stats_tags['Content_Warnings']['n_registros']}/{n_obras} obras = {stats_tags['Content_Warnings']['cobertura_%']:.1f}%)",
                     loc="left", fontsize=10, fontweight="bold")
        salvar_fig(fig, "fig_06_content_warnings", formatos, out_dir)

    # FIG 7: Metodo de resolucao da ponte (barras)
    if stats_cat.get("fonte_match"):
        fm = pd.Series(stats_cat["fonte_match"]).sort_values()
        fig, ax = plt.subplots(figsize=(W1 + 1.0, 2.5))
        ax.barh(fm.index, fm.values, color=COR_PRIM, edgecolor="black", linewidth=0.5)
        for y, v in enumerate(fm.values):
            ax.text(v, y, f" {v} ({100 * v / fm.sum():.1f}%)".replace(".", ","), va="center", fontsize=9, fontweight="medium")
        ax.set_xlim(0, fm.max() * 1.32)
        ax.set_xlabel(f"Nº de edições (total = {int(fm.sum())})")
        ax.grid(axis="y", visible=False)
        titulo_ax(ax, "Método de resolução da ponte", fig_titulos)
        salvar_fig(fig, "fig_07_fonte_match", formatos, out_dir)

    # FIG 8: Score_Ponte por metodo (com mediana e n anotados)
    if {"Score_Ponte", "Metodo_Ponte"} <= set(df.columns):
        d = df.dropna(subset=["Score_Ponte", "Metodo_Ponte"])
        if len(d):
            ordem = d.groupby("Metodo_Ponte")["Score_Ponte"].median().sort_values().index.tolist()
            fig, ax = plt.subplots(figsize=(W1 + 1.2, 2.8))
            sns.boxplot(data=d, y="Metodo_Ponte", x="Score_Ponte", order=ordem, color=COR_PRIM, width=0.5,
                        fliersize=0, boxprops=dict(alpha=0.4), ax=ax)
            sns.stripplot(data=d, y="Metodo_Ponte", x="Score_Ponte", order=ordem, color="black", size=3.0,
                          alpha=0.6, jitter=0.2, ax=ax)
            for y, m in enumerate(ordem):
                sub = d.loc[d["Metodo_Ponte"] == m, "Score_Ponte"]
                ax.text(1.02, y, f"med={sub.median():.2f} (n={len(sub)})".replace(".", ","),
                        va="center", ha="left", fontsize=8.5)
            ax.set_xlabel("Score da ponte")
            ax.set_ylabel("")
            ax.set_xlim(0, 1.32)
            titulo_ax(ax, "Score da ponte por método", fig_titulos)
            salvar_fig(fig, "fig_08_score_ponte", formatos, out_dir)

    # FIG 9: Paginas (com numeros nas barras)
    if stats_dist.get("paginas"):
        p = df.loc[df["Num_Paginas_Clean"] > 0, "Num_Paginas_Clean"].dropna()
        fig, ax = plt.subplots(figsize=(W1 + 1.0, 3.2))
        counts, edges, _ = ax.hist(p, bins=16, color=COR_PRIM, edgecolor="black", linewidth=0.5)
        centers = 0.5 * (edges[:-1] + edges[1:])
        _anotar_barras_verticais(ax, centers, counts, fontsize=8)
        ax.axvline(p.mean(), color=COR_TERC, ls=":", lw=1.4, label=f"Média = {p.mean():.0f} pág.")
        ax.axvline(p.median(), color=COR_SEC, ls="--", lw=1.4, label=f"Mediana = {p.median():.0f} pág. (n={len(p)})")
        ax.set_xlabel("Nº de páginas (edição PT-BR)")
        ax.set_ylabel("Nº de edições")
        ax.legend()
        titulo_ax(ax, "Distribuição de páginas", fig_titulos)
        salvar_fig(fig, "fig_09_paginas", formatos, out_dir)

    # FIG 10: Spice_Status / Spice_Nivel (com numeros em AMBOS os paineis)
    if "Spice_Status" in ob.columns:
        fig, axes = plt.subplots(1, 2, figsize=(W2, 3.2))
        vc = ob["Spice_Status"].value_counts()
        cores = [COR_CINZA if k == "sem_sinal" else COR_SEC for k in vc.index]
        xs0 = np.arange(len(vc))
        axes[0].bar(xs0, vc.values, color=cores, edgecolor="black", linewidth=0.5)
        axes[0].set_xticks(xs0)
        axes[0].set_xticklabels(vc.index)
        _anotar_barras_verticais(axes[0], xs0, vc.values, total=int(vc.sum()), mostrar_pct=True, fontsize=8.5)
        axes[0].set_ylabel("Nº de obras")
        axes[0].set_title(f"(a) Spice_Status (n = {int(vc.sum())})", loc="left", fontsize=10, fontweight="bold")
        axes[0].grid(axis="x", visible=False)
        if "Spice_Nivel" in ob.columns:
            niv = pd.to_numeric(ob["Spice_Nivel"], errors="coerce").dropna().astype(int).value_counts().reindex([0, 1, 2, 3], fill_value=0)
            xs1 = np.arange(4)
            axes[1].bar(xs1, niv.values, color=COR_SEC, edgecolor="black", linewidth=0.5)
            axes[1].set_xticks(xs1)
            axes[1].set_xticklabels(["0\nfechado", "1\nsensual", "2\ncena aberta", "3\nexplícito"])
            _anotar_barras_verticais(axes[1], xs1, niv.values, total=max(int(niv.sum()), 1), mostrar_pct=True, fontsize=8.5)
            axes[1].set_ylabel("Nº de obras (com sinal)")
            axes[1].set_title(f"(b) Spice_Nivel (n com sinal = {int(niv.sum())})", loc="left", fontsize=10, fontweight="bold")
            axes[1].grid(axis="x", visible=False)
        salvar_fig(fig, "fig_10_spice", formatos, out_dir)

    # FIG 11: Cobertura de tags por quartil de popularidade (com % e n claros)
    cob = cobertura_por_popularidade(ob)
    if cob is not None:
        fig, ax = plt.subplots(figsize=(W1 + 1.6, 3.2))
        c2 = cob.copy()
        c2.columns = [c.replace("_Hardcover", "").replace("_", " ") for c in c2.columns]
        annot_cob = (c2 * 100).round(1).astype(str).map(lambda s: s.replace(".", ",") + "%")
        sns.heatmap(c2, annot=annot_cob, fmt="", cmap=CMAP_SEQ, vmin=0, vmax=1, linewidths=0.6, linecolor="white",
                    cbar_kws={"label": "Fração de obras com ≥1 tag"}, ax=ax, annot_kws={"fontsize": 9, "fontweight": "bold"})
        ax.set_ylabel("Quartil de popularidade")
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        salvar_fig(fig, "fig_11_cobertura_popularidade", formatos, out_dir)

    # FIG 12: Conteudo Sensivel Geral e Spice positivo por quartil de popularidade (com % e k/n explicitos)
    sp = extra.get("spice_por_pop")
    sn = extra.get("sensivel_por_pop")
    if (sp is not None and len(sp)) or (sn is not None and len(sn)):
        fig, axes = plt.subplots(1, 2, figsize=(W2 + 0.6, 3.4), sharey=True)
        paineis = [
            (axes[0], sn, "k_sensivel_alto", COR_PRIM, "(a) Conteúdo Sensível Geral (alto)"),
            (axes[1], sp, "k_positivo", COR_SEC, "(b) Apenas Eixo Sexual (Spice positivo)"),
        ]
        for ax, tab, col_k, cor, tit in paineis:
            if tab is None or not len(tab):
                ax.axis("off")
                continue
            xs = np.arange(len(tab))
            ax.errorbar(xs, tab["p"], yerr=[tab["p"] - tab["lo"], tab["hi"] - tab["p"]],
                        fmt="o-", color=cor, ecolor="black", capsize=4, markersize=6, lw=1.2)
            ax.set_xticks(xs)
            ax.set_xticklabels(tab["Quartil"])
            for x, (_, row) in enumerate(tab.iterrows()):
                p_val = float(row["p"])
                k_val = int(row[col_k])
                n_val = int(row["n"])
                txt = f"{100 * p_val:.1f}%\n({k_val}/{n_val})".replace(".", ",")
                y_pos = min(0.92, float(row["hi"]) + 0.04)
                ax.text(x, y_pos, txt, fontsize=8.5, ha="center", va="bottom", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
            ax.set_ylim(-0.02, 1.08)
            ax.set_xlabel("Quartil de popularidade (leitores)")
            ax.set_title(tit, loc="left", fontsize=10, fontweight="bold")
        axes[0].set_ylabel("Proporção de obras (IC de Wilson 95%)")
        salvar_fig(fig, "fig_12_spice_vs_popularidade", formatos, out_dir)

    # FIG 13: Tipo editorial × Sensivel_Label e × Spice_Status
    col_lbl, col_conf, _ = resolver_colunas_rotulo(ob)
    if "Tipo_Editorial_Editora" in ob.columns and ("Spice_Status" in ob.columns or col_lbl in ob.columns):
        col_alvo = col_lbl if col_lbl in ob.columns else "Spice_Status"
        ct = pd.crosstab(ob["Tipo_Editorial_Editora"].fillna("NA"), ob[col_alvo])
        prop = ct.div(ct.sum(axis=1), axis=0)
        annot = ct.astype(str) + " (" + (prop * 100).round(1).astype(str).map(lambda s: s.replace(".", ",")) + "%)"
        fig, ax = plt.subplots(figsize=(W1 + 1.8, 3.2))
        sns.heatmap(prop, annot=annot, fmt="", cmap=CMAP_SEQ, vmin=0, vmax=1, linewidths=0.6, linecolor="white",
                    cbar_kws={"label": "Proporção por tipo editorial"}, ax=ax, annot_kws={"fontsize": 8.5, "fontweight": "bold"})
        ax.set_xlabel(f"Rótulo ({col_alvo})")
        ax.set_ylabel("Tipo editorial")
        salvar_fig(fig, "fig_13_tipo_editorial_spice", formatos, out_dir)

    # FIG 14: Rotulo triangulado (Sensivel_Label v3.1), empilhado por confianca COM numeros em cada segmento e no topo
    if {col_lbl, col_conf} <= set(ob.columns):
        ordem_lbl = [k for k in ("alto", "provavel_baixo", "baixo", "indeterminado") if k in ob[col_lbl].values or k in ("alto", "provavel_baixo", "indeterminado")]
        ct = pd.crosstab(ob[col_lbl], ob[col_conf]).reindex(ordem_lbl).fillna(0).astype(int)
        cores = {
            "forte": OKABE_ITO[0],
            "fraca": OKABE_ITO[4],
            "conflito": OKABE_ITO[3],
            "circular": OKABE_ITO[1],
            "nao_validado": OKABE_ITO[5],
            "sem_evidencia": COR_CINZA,
        }
        fig, ax = plt.subplots(figsize=(W1 + 1.8, 3.8))
        xs = np.arange(len(ct))
        base = np.zeros(len(ct))
        totais_barra = ct.sum(axis=1).values
        vmax = max(totais_barra) if len(totais_barra) else 1

        for col in ct.columns:
            vals = ct[col].values
            ax.bar(xs, vals, bottom=base, label=f"{col} (n={int(vals.sum())})",
                   color=cores.get(col, COR_CINZA), edgecolor="black", linewidth=0.5, width=0.6)
            for i, (b, v) in enumerate(zip(base, vals)):
                if v >= 5:
                    ax.text(xs[i], b + v / 2, f"{int(v)}", ha="center", va="center",
                            fontsize=8.5, fontweight="bold", color="white" if col in ("forte", "sem_evidencia") else "black")
            base += vals

        # Anota o total e % no topo de cada barra empilhada
        for i, tot in enumerate(totais_barra):
            pct = 100 * tot / max(n_obras, 1)
            ax.text(xs[i], tot + vmax * 0.02, f"Total: {int(tot)}\n({pct:.1f}%)".replace(".", ","),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(xs)
        ax.set_xticklabels(ct.index, fontsize=9.5, fontweight="medium")
        ax.set_ylim(0, vmax * 1.25)
        ax.set_ylabel(f"Nº de obras (total = {n_obras})")
        ax.set_title("Distribuição do Rótulo Triangulado (Sensivel_Label) por Confiança", loc="left", fontsize=10, fontweight="bold")
        ax.legend(title="Confiança", loc="upper left", bbox_to_anchor=(1.01, 1.0))
        ax.grid(axis="x", visible=False)
        salvar_fig(fig, "fig_14_spice_label", formatos, out_dir)

    # FIG 15: Prevalencia por Eixo de Conteudo Sensivel e Problematico (com IC de Wilson 95% e contagens)
    prev = prevalencias_flags(df)
    mapa_eixos = {
        "Flag_Qualquer_Eixo_Sensivel": "Qualquer Eixo Sensível (União)",
        "Flag_Eixo_Sexo": "1. Sexo / Erotismo / Spice",
        "Flag_Eixo_Violencia": "2. Violência / Crime / Gore",
        "Flag_Eixo_Abuso_Gatilhos": "3. Abuso / Rel. Tóxico / Saúde Mental",
        "Flag_Eixo_Drogas_Alcool": "4. Drogas / Bebidas Alcoólicas",
    }
    sub_eixos = prev[prev["Flag"].isin(mapa_eixos.keys())].copy()
    if len(sub_eixos):
        sub_eixos["Rotulo"] = sub_eixos["Flag"].map(mapa_eixos)
        sub_eixos = sub_eixos.sort_values("prop", ascending=True).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(W2, 3.4))
        ys = np.arange(len(sub_eixos))
        cores_eixos = [COR_SEC if f == "Flag_Qualquer_Eixo_Sensivel" else COR_PRIM for f in sub_eixos["Flag"]]
        ax.barh(ys, sub_eixos["prop"] * 100, color=cores_eixos, edgecolor="black", linewidth=0.5, alpha=0.88)
        err_lo = (sub_eixos["prop"] - sub_eixos["IC95_inf"]) * 100
        err_hi = (sub_eixos["IC95_sup"] - sub_eixos["prop"]) * 100
        ax.errorbar(sub_eixos["prop"] * 100, ys, xerr=[err_lo, err_hi], fmt="none", ecolor="black", capsize=4, lw=1.2)
        for y, (_, r) in enumerate(sub_eixos.iterrows()):
            x_txt = float(r["IC95_sup"]) * 100 + 1.0
            txt = f"{int(r['k'])}/{int(r['n'])} ({100 * r['prop']:.1f}% | IC95%: [{100 * r['IC95_inf']:.1f}%, {100 * r['IC95_sup']:.1f}%])".replace(".", ",")
            ax.text(x_txt, y, txt, va="center", ha="left", fontsize=8.5, fontweight="bold")
        ax.set_yticks(ys)
        ax.set_yticklabels(sub_eixos["Rotulo"], fontsize=9.5)
        ax.set_xlim(0, max(sub_eixos["IC95_sup"] * 100) * 1.65)
        ax.set_xlabel(f"Prevalência nas obras (% de {n_obras} obras, com IC de Wilson 95%)")
        ax.set_title("Prevalência por Eixo de Conteúdo Sensível e Problemático (Multi-Label)", loc="left", fontsize=10.5, fontweight="bold")
        ax.grid(axis="y", visible=False)
        salvar_fig(fig, "fig_15_eixos_sensiveis", formatos, out_dir)
