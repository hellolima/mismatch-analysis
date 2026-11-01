"""Avaliacao de robustez do rotulo triangulado (rodar DEPOIS de ponte; recalcula a rotulagem internamente).

Experimentos (nomes seguem a auditoria do projeto):
  E3  Funil de atrito: seed -> resolvidas -> escopo -> obras -> rotuladas, por canal de origem
  E4  Ablacao de fontes: rotulo sem cada uma das 5 fontes (tags, vocab, sinopse, maturity, cw)
  E5  Sensibilidade: pop_pct x exigir_cw (definicao de Cobertura_Boa) x min_count (via Spice_Status)
  E6  Concordancia entre fontes: kappa de Fleiss (todas) e Cohen par-a-par, so em obras com >= 2 votos
  E7  Coocorrencia dos 4 eixos: Jaccard par-a-par + distribuicao do numero de eixos por obra
  E9  Omissao editorial: entre obras com CW sensivel (V_cw=alto), % em que a sinopse PT nao avisa

Saidas:
  outputs/reports/avaliacao_rotulo.txt
  outputs/tables/aval_funil.csv, aval_ablacao.csv, aval_sensibilidade.csv, aval_kappa_fontes.csv,
                 aval_coocorrencia_jaccard.csv, aval_n_eixos.csv, aval_omissao_editorial.csv,
                 regras_decisao.csv, lexico_eixos.csv
  outputs/figures/fig_16_funil.png, fig_17_ablacao.png, fig_18_coocorrencia.png, fig_19_omissao_editorial.png

Uso: python scripts/pipeline_ptbr/avaliar_rotulo.py [--ano-min 2015] [--pop-pct 50]
"""
import os
import sys
import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_ptbr import config as C                                     # noqa: E402
from pipeline_ptbr.dataset import filtrar_escopo, deduplicar_por_obra     # noqa: E402
from pipeline_ptbr.rotulagem import rotular, tabela_regras, FONTES         # noqa: E402
from pipeline_ptbr.lexicon import exportar_lexico_csv, ORDEM_EIXOS, COLUNA_EIXO, EIXOS, RE_POR_EIXO  # noqa: E402
from pipeline_ptbr.eda_stats import wilson_ic                             # noqa: E402
from pipeline_ptbr.common import sem_acento                               # noqa: E402

ROTULOS = ("alto", "baixo", "provavel_baixo", "indeterminado")


# ---------------------------------------------------------------------------
# Estatistica de concordancia (sem dependencias externas)
# ---------------------------------------------------------------------------
def cohen_kappa(a: pd.Series, b: pd.Series) -> Tuple[float, int]:
    """Kappa de Cohen entre duas series categoricas (ignora pares com NaN). Devolve (kappa, n)."""
    m = pd.concat([a, b], axis=1).dropna()
    n = len(m)
    if n == 0:
        return (np.nan, 0)
    x, y = m.iloc[:, 0].astype(str), m.iloc[:, 1].astype(str)
    po = float((x == y).mean())
    cats = sorted(set(x) | set(y))
    pe = sum((x == c).mean() * (y == c).mean() for c in cats)
    if pe >= 1.0:
        return (1.0 if po == 1.0 else 0.0, n)
    return ((po - pe) / (1 - pe), n)


def fleiss_kappa(tabela: pd.DataFrame) -> Tuple[float, int]:
    """Kappa de Fleiss. `tabela`: linhas = itens, colunas = categorias, valores = nº de avaliadores
    que escolheram a categoria. Exige o MESMO numero de avaliadores por item (linhas com soma diferente
    da moda sao descartadas). Devolve (kappa, n_itens_usados)."""
    if tabela.empty:
        return (np.nan, 0)
    soma = tabela.sum(axis=1)
    k = int(soma.mode().iloc[0])
    t = tabela[soma == k]
    n = len(t)
    if n == 0 or k < 2:
        return (np.nan, n)
    p_j = t.sum(axis=0) / (n * k)
    p_i = ((t ** 2).sum(axis=1) - k) / (k * (k - 1))
    p_bar, pe = float(p_i.mean()), float((p_j ** 2).sum())
    if pe >= 1.0:
        return (1.0 if p_bar == 1.0 else 0.0, n)
    return ((p_bar - pe) / (1 - pe), n)


def jaccard(a: pd.Series, b: pd.Series) -> float:
    a, b = a.astype(bool), b.astype(bool)
    u = int((a | b).sum())
    return float((a & b).sum() / u) if u else np.nan


# ---------------------------------------------------------------------------
# E3: funil de atrito
# ---------------------------------------------------------------------------
def funil_atrito(seed: pd.DataFrame, livros: pd.DataFrame, rotulado_ob: pd.DataFrame,
                 meta: Dict[str, int]) -> pd.DataFrame:
    def por_canal(df: pd.DataFrame, col: str) -> Dict[str, int]:
        if col not in df.columns:
            return {}
        s = df[col].fillna("(sem)").astype(str).str.split(r" \| ").explode().str.strip()
        return s.value_counts().to_dict()

    etapas = [
        ("1. Edicoes no seed (Google Books, ISBN BR)", len(seed), por_canal(seed, "Tipos_Origem")),
        ("2. Resolvidas na ponte (edicao -> obra Hardcover)", len(livros), por_canal(livros, "Tipos_Origem")),
        (f"3. Em escopo (ano >= {meta.get('ano_min')}, sem formato excluido)", meta.get("n_final", 0), {}),
        ("4. Obras unicas (dedup FRBR)", len(rotulado_ob), por_canal(rotulado_ob, "Tipos_Origem")),
        ("5. Rotuladas (alto | baixo | provavel_baixo)",
         int(rotulado_ob["Sensivel_Label"].isin(["alto", "baixo", "provavel_baixo"]).sum()),
         por_canal(rotulado_ob[rotulado_ob["Sensivel_Label"] != "indeterminado"], "Tipos_Origem")),
        ("5a.   das quais 'alto'", int((rotulado_ob["Sensivel_Label"] == "alto").sum()),
         por_canal(rotulado_ob[rotulado_ob["Sensivel_Label"] == "alto"], "Tipos_Origem")),
    ]
    canais = sorted({c for _, _, d in etapas for c in d})
    rows = []
    base = etapas[0][1]
    for nome, n, d in etapas:
        row = {"Etapa": nome, "N": n, "Pct_do_seed": round(100 * n / base, 1) if base else np.nan}
        row.update({f"canal:{c}": d.get(c, "") for c in canais})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# E4: ablacao de fontes
# ---------------------------------------------------------------------------
def ablacao_fontes(df_escopo: pd.DataFrame, pop_pct: float) -> pd.DataFrame:
    base = deduplicar_por_obra(rotular(df_escopo, pop_pct))
    ref = base["Sensivel_Label"]
    rows = []

    def linha(nome: str, ob: pd.DataFrame) -> dict:
        vc = ob["Sensivel_Label"].value_counts()
        forte = int((ob["Sensivel_Label_Confianca"] == "forte").sum())
        alinhado = ob.set_index("Hardcover_Book_ID")["Sensivel_Label"]
        ref_al = base.set_index("Hardcover_Book_ID")["Sensivel_Label"]
        comum = alinhado.index.intersection(ref_al.index)
        mudou = int((alinhado.loc[comum] != ref_al.loc[comum]).sum())
        return {"Configuracao": nome, **{r: int(vc.get(r, 0)) for r in ROTULOS},
                "alto_forte": forte, "obras_que_mudaram_vs_completo": mudou}

    rows.append(linha("completo (5 fontes)", base))
    for f in FONTES:
        rows.append(linha(f"sem {f}", deduplicar_por_obra(rotular(df_escopo, pop_pct, ignorar=(f,)))))
    rows.append(linha("so comunidade (tags+vocab)", deduplicar_por_obra(rotular(df_escopo, pop_pct, ignorar=("sinopse", "maturity", "cw")))))
    rows.append(linha("so sinopse", deduplicar_por_obra(rotular(df_escopo, pop_pct, ignorar=("tags", "vocab", "maturity", "cw")))))
    rows.append(linha("so cw", deduplicar_por_obra(rotular(df_escopo, pop_pct, ignorar=("tags", "vocab", "sinopse", "maturity")))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# E5: sensibilidade a pop_pct e a definicao de Cobertura_Boa
# ---------------------------------------------------------------------------
def sensibilidade(df_escopo: pd.DataFrame, pcts=(25, 35, 50, 65, 75)) -> pd.DataFrame:
    rows = []
    for exigir_cw in (False, True):
        for p in pcts:
            ob = deduplicar_por_obra(rotular(df_escopo, float(p), exigir_cw=exigir_cw))
            vc = ob["Sensivel_Label"].value_counts()
            rows.append({"pop_pct": p, "Cobertura_Boa_exige_CW": exigir_cw,
                         "limiar_leitores": int(ob.attrs.get("pop_limiar", np.nan)) if "pop_limiar" in ob.attrs else np.nan,
                         "n_Cobertura_Boa": int(ob["Cobertura_Boa"].sum()),
                         **{r: int(vc.get(r, 0)) for r in ROTULOS}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# E6: concordancia entre fontes
# ---------------------------------------------------------------------------
def concordancia_fontes(ob: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    v = ob[[f"V_{f}" for f in FONTES]].copy()
    # Cohen par-a-par (so obras onde AMBAS votaram)
    pares = []
    for a, b in itertools.combinations(FONTES, 2):
        k, n = cohen_kappa(v[f"V_{a}"], v[f"V_{b}"])
        ambos = v[[f"V_{a}", f"V_{b}"]].dropna()
        conc = float((ambos.iloc[:, 0] == ambos.iloc[:, 1]).mean()) if len(ambos) else np.nan
        n_cats = len(set(ambos.values.ravel())) if len(ambos) else 0
        pares.append({"Fonte_A": a, "Fonte_B": b, "n_obras_com_ambos": n,
                      "concordancia_bruta": round(conc, 3) if n else np.nan,
                      "kappa_cohen": round(k, 3) if n else np.nan,
                      "n_categorias_observadas": n_cats,
                      "nota": "degenerado: so 1 categoria observada ('alto'); kappa nao informativo" if n and n_cats <= 1 else ""})
    # Fleiss sobre obras com >= 2 votos, usando categorias alto/baixo (abstencao nao conta como voto)
    cats = ["alto", "baixo", "conflito"]
    tab = pd.DataFrame({c: (v == c).sum(axis=1) for c in cats})
    n_votos = tab.sum(axis=1)
    resumo = []
    for k_min in (2, 3):
        sub = tab[n_votos >= k_min]
        # Fleiss exige k constante: avaliamos por estrato de nº de votos e reportamos cada um
        for k, g in sub.groupby(n_votos[n_votos >= k_min]):
            kap, n = fleiss_kappa(g)
            resumo.append({"n_votos_por_obra": int(k), "n_obras": int(n), "kappa_fleiss": round(kap, 3) if n else np.nan,
                           "prop_unanime_alto": round(float((g["alto"] == k).mean()), 3) if n else np.nan})
        break
    return pd.DataFrame(pares), pd.DataFrame(resumo).drop_duplicates()


# ---------------------------------------------------------------------------
# E7: coocorrencia dos eixos
# ---------------------------------------------------------------------------
def coocorrencia_eixos(ob: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = [COLUNA_EIXO[e] for e in ORDEM_EIXOS]
    X = ob[cols].fillna(False).astype(bool)
    jac = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            jac.loc[a, b] = 1.0 if a == b else jaccard(X[a], X[b])
    n_eixos = X.sum(axis=1)
    dist = n_eixos.value_counts().sort_index().rename_axis("n_eixos").reset_index(name="n_obras")
    dist["pct_obras"] = (100 * dist["n_obras"] / len(ob)).round(1)
    combos = (X.apply(lambda r: " + ".join(c.replace("Flag_Eixo_", "") for c in cols if r[c]) or "(nenhum)", axis=1)
              .value_counts().rename_axis("Combinacao").reset_index(name="n_obras"))
    return jac, dist, combos


# ---------------------------------------------------------------------------
# E9: omissao editorial (a sinopse PT avisa o que a comunidade avisa?)
# ---------------------------------------------------------------------------
def omissao_editorial(ob: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sin = ob["Sinopse_PT"].map(sem_acento) if "Sinopse_PT" in ob.columns else pd.Series("", index=ob.index)
    cw = ob["Content_Warnings"].map(sem_acento) if "Content_Warnings" in ob.columns else pd.Series("", index=ob.index)
    for e in ORDEM_EIXOS:
        tem_cw = cw.map(lambda s: bool(RE_POR_EIXO[e]["cw"].search(s)))
        avisa_pt = sin.map(lambda s: bool(RE_POR_EIXO[e]["pt"].search(s)))
        n = int(tem_cw.sum())
        k_omite = int((tem_cw & ~avisa_pt).sum())
        p, lo, hi = wilson_ic(k_omite, n)
        rows.append({"Eixo": EIXOS[e]["rotulo"], "obras_com_CW_comunitario": n,
                     "sinopse_PT_nao_menciona": k_omite, "taxa_omissao": round(p, 3) if n else np.nan,
                     "IC95_inf": round(lo, 3) if n else np.nan, "IC95_sup": round(hi, 3) if n else np.nan})
    # geral: V_cw=alto e V_sinopse abstem
    if {"V_cw", "V_sinopse"} <= set(ob.columns):
        base = ob[ob["V_cw"] == "alto"]
        n = len(base)
        k = int(base["V_sinopse"].isna().sum())
        p, lo, hi = wilson_ic(k, n)
        rows.append({"Eixo": "QUALQUER (V_cw=alto e V_sinopse abstem)", "obras_com_CW_comunitario": n,
                     "sinopse_PT_nao_menciona": k, "taxa_omissao": round(p, 3) if n else np.nan,
                     "IC95_inf": round(lo, 3) if n else np.nan, "IC95_sup": round(hi, 3) if n else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figuras (opcionais: exigem matplotlib)
# ---------------------------------------------------------------------------
def _figuras(funil: pd.DataFrame, abl: pd.DataFrame, jac: pd.DataFrame, dist: pd.DataFrame,
             omis: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from pipeline_ptbr.eda_plots import aplicar_estilo, salvar_fig, OKABE_ITO, COR_PRIM, COR_SEC, COR_CINZA
    except ImportError as e:
        print(f"[avaliacao] figuras ignoradas ({e})")
        return
    aplicar_estilo()

    # fig 16: funil
    f = funil[~funil["Etapa"].str.startswith("5a")]
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ys = np.arange(len(f))[::-1]
    ax.barh(ys, f["N"], color=COR_PRIM, edgecolor="black", linewidth=0.5)
    for y, (n, p) in zip(ys, zip(f["N"], f["Pct_do_seed"])):
        ax.text(n, y, f"  {int(n)} ({p:.1f}% do seed)".replace(".", ","), va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels(f["Etapa"], fontsize=9)
    ax.set_xlim(0, f["N"].max() * 1.45)
    ax.set_xlabel("Nº de registros")
    ax.set_title("Funil de atrito: do seed ao rótulo", loc="left", fontweight="bold")
    ax.grid(axis="y", visible=False)
    salvar_fig(fig, "fig_16_funil", ("png", "pdf"), out_dir)

    # fig 17: ablacao
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    xs = np.arange(len(abl))
    base = np.zeros(len(abl))
    cores = {"alto": OKABE_ITO[1], "baixo": OKABE_ITO[2], "provavel_baixo": OKABE_ITO[5], "indeterminado": COR_CINZA}
    for r in ROTULOS:
        vals = abl[r].values
        ax.bar(xs, vals, bottom=base, label=r, color=cores[r], edgecolor="black", linewidth=0.5, width=0.65)
        for i, (b, v) in enumerate(zip(base, vals)):
            if v >= 6:
                ax.text(xs[i], b + v / 2, f"{int(v)}", ha="center", va="center", fontsize=8.5, fontweight="bold",
                        color="white" if r == "indeterminado" else "black")
        base += vals
    for i, m in enumerate(abl["obras_que_mudaram_vs_completo"]):
        ax.text(xs[i], base[i] + 3, f"Δ={int(m)}", ha="center", fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(abl["Configuracao"], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Nº de obras")
    ax.set_ylim(0, base.max() * 1.15)
    ax.set_title("Ablação de fontes: distribuição do rótulo ao remover cada fonte (Δ = obras que mudam)", loc="left", fontweight="bold", fontsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(axis="x", visible=False)
    salvar_fig(fig, "fig_17_ablacao", ("png", "pdf"), out_dir)

    # fig 18: coocorrencia
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1.2, 1]})
    j = jac.copy()
    j.index = [c.replace("Flag_Eixo_", "") for c in j.index]
    j.columns = j.index
    sns.heatmap(j.astype(float), annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=axes[0],
                linewidths=0.6, linecolor="white", cbar_kws={"label": "Jaccard"}, annot_kws={"fontsize": 9, "fontweight": "bold"})
    axes[0].set_title("(a) Sobreposição entre eixos (Jaccard)", loc="left", fontweight="bold", fontsize=10)
    xs = np.arange(len(dist))
    axes[1].bar(xs, dist["n_obras"], color=COR_SEC, edgecolor="black", linewidth=0.5)
    for x, (n, p) in enumerate(zip(dist["n_obras"], dist["pct_obras"])):
        axes[1].text(x, n + 2, f"{int(n)}\n({p:.1f}%)".replace(".", ","), ha="center", fontsize=8.5, fontweight="bold")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(dist["n_eixos"])
    axes[1].set_xlabel("Nº de eixos sensíveis presentes na obra")
    axes[1].set_ylabel("Nº de obras")
    axes[1].set_ylim(0, dist["n_obras"].max() * 1.3)
    axes[1].set_title("(b) Obras por nº de eixos", loc="left", fontweight="bold", fontsize=10)
    axes[1].grid(axis="x", visible=False)
    salvar_fig(fig, "fig_18_coocorrencia", ("png", "pdf"), out_dir)

    # fig 19: omissao editorial
    o = omis.dropna(subset=["taxa_omissao"])
    if len(o):
        fig, ax = plt.subplots(figsize=(8.6, 3.2))
        ys = np.arange(len(o))
        ax.barh(ys, o["taxa_omissao"] * 100, color=COR_PRIM, edgecolor="black", linewidth=0.5)
        ax.errorbar(o["taxa_omissao"] * 100, ys,
                    xerr=[(o["taxa_omissao"] - o["IC95_inf"]) * 100, (o["IC95_sup"] - o["taxa_omissao"]) * 100],
                    fmt="none", ecolor="black", capsize=4)
        for y, (_, r) in enumerate(o.iterrows()):
            ax.text(r["IC95_sup"] * 100 + 1, y,
                    f"{int(r['sinopse_PT_nao_menciona'])}/{int(r['obras_com_CW_comunitario'])} ({100 * r['taxa_omissao']:.1f}%)".replace(".", ","),
                    va="center", fontsize=9, fontweight="bold")
        ax.set_yticks(ys)
        ax.set_yticklabels(o["Eixo"], fontsize=9)
        ax.set_xlim(0, 135)
        ax.set_xlabel("% de obras com content warning comunitário cuja sinopse PT-BR NÃO menciona o tema (IC Wilson 95%)")
        ax.set_title("Omissão editorial: o que a comunidade avisa e a sinopse brasileira não", loc="left", fontweight="bold", fontsize=10)
        ax.grid(axis="y", visible=False)
        salvar_fig(fig, "fig_19_omissao_editorial", ("png", "pdf"), out_dir)


# ---------------------------------------------------------------------------
def executar_avaliacao(ano_min: int = C.ANO_MIN_PADRAO, pop_pct: float = C.POP_PCT_PADRAO) -> str:
    C.garantir_dirs()
    T, R, F = Path(C.OUT_TABLES), Path(C.OUT_REPORTS), Path(C.OUT_FIGURES)
    seed = pd.read_csv(C.SEED_CSV, dtype=str) if os.path.exists(C.SEED_CSV) else pd.DataFrame()
    livros = pd.read_csv(C.OUT_LIVROS)
    df_escopo, meta = filtrar_escopo(livros, ano_min)
    rot = rotular(df_escopo, pop_pct)
    ob = deduplicar_por_obra(rot).reset_index(drop=True)

    funil = funil_atrito(seed, livros, ob, meta)
    abl = ablacao_fontes(df_escopo, pop_pct)
    sens = sensibilidade(df_escopo)
    kap_pares, kap_fleiss = concordancia_fontes(ob)
    jac, dist, combos = coocorrencia_eixos(ob)
    omis = omissao_editorial(ob)
    regras = tabela_regras()
    lex = exportar_lexico_csv(T / "lexico_eixos.csv")

    for nome, t in (("aval_funil", funil), ("aval_ablacao", abl), ("aval_sensibilidade", sens),
                    ("aval_kappa_fontes", kap_pares), ("aval_kappa_fleiss", kap_fleiss),
                    ("aval_coocorrencia_jaccard", jac), ("aval_n_eixos", dist), ("aval_combinacoes_eixos", combos),
                    ("aval_omissao_editorial", omis), ("regras_decisao", regras)):
        t.to_csv(T / f"{nome}.csv", index=(nome == "aval_coocorrencia_jaccard"), encoding="utf-8")

    L: List[str] = []
    w = L.append
    w("AVALIAÇÃO DE ROBUSTEZ DO RÓTULO TRIANGULADO (Sensivel_Label)\n" + "=" * 90)
    w(f"Obras: {len(ob)} | ano >= {ano_min} | pop_pct = {pop_pct:.0f}\n")
    w("E3) FUNIL DE ATRITO (seed -> rótulo), por canal de origem\n" + funil.to_string(index=False) + "\n")
    w("E4) ABLAÇÃO DE FONTES (Δ = obras cujo rótulo muda em relação à configuração completa)\n" + abl.to_string(index=False) + "\n")
    w("E5) SENSIBILIDADE: pop_pct × definição de Cobertura_Boa (exige CW = definição antiga)\n" + sens.to_string(index=False) + "\n")
    w("E6) CONCORDÂNCIA ENTRE FONTES\n  Cohen par-a-par (só obras onde ambas votaram):\n" + kap_pares.to_string(index=False)
      + "\n  Fleiss por nº de votos por obra:\n" + kap_fleiss.to_string(index=False)
      + "\n  Leitura: como quase não há voto 'baixo', a concordância bruta é alta e o kappa é degenerado "
        "(sem variância de categoria). Isso É o achado: o sistema é Positive-Unlabeled.\n")
    w("E7) COOCORRÊNCIA DOS 4 EIXOS\n  Jaccard:\n" + jac.round(2).to_string() + "\n  Nº de eixos por obra:\n" + dist.to_string(index=False)
      + "\n  Combinações mais frequentes:\n" + combos.head(12).to_string(index=False) + "\n")
    w("E9) OMISSÃO EDITORIAL (a sinopse PT-BR menciona o que a comunidade sinaliza?)\n" + omis.to_string(index=False) + "\n")
    w("REGRAS DE DECISÃO (ordem = prioridade)\n" + regras.to_string(index=False) + "\n")
    w(f"Léxico exportado: {T / 'lexico_eixos.csv'} ({len(lex)} termos)")
    txt = "\n".join(L)
    (R / "avaliacao_rotulo.txt").write_text(txt, encoding="utf-8")
    print(txt)

    _figuras(funil, abl, jac, dist, omis, F)
    return txt


def main() -> None:
    ap = argparse.ArgumentParser(description="Avaliação de robustez do rótulo (ablação, sensibilidade, kappa, funil, coocorrência, omissão)")
    ap.add_argument("--ano-min", type=int, default=C.ANO_MIN_PADRAO)
    ap.add_argument("--pop-pct", type=float, default=C.POP_PCT_PADRAO)
    a = ap.parse_args()
    executar_avaliacao(a.ano_min, a.pop_pct)


if __name__ == "__main__":
    main()
