"""Gold Standard: planilhas de anotacao humana cega (por eixo) e avaliacao do rotulo e da ponte.

Duas fases:
  gerar   -> cria as planilhas para preencher (SEM o rotulo automatico, para a anotacao ser cega):
             data/gold_standard/anotacao_eixos_cega.csv      (96 obras: 1 linha por obra, colunas A_* e B_*)
             data/gold_standard/anotacao_eixos_chave.csv     (gabarito automatico; NAO abrir antes de anotar)
             data/gold_standard/auditoria_ponte_amostra.csv  (amostra estratificada por metodo p/ Confere_Manual)
             data/gold_standard/PROTOCOLO_ANOTACAO.md        (instrucoes para os anotadores)
  avaliar -> le as planilhas preenchidas e calcula:
             * Cohen's kappa entre anotadores A e B (geral e por eixo) + % de concordancia bruta
             * Precisao / recall / F1 do rotulo automatico (Sensivel_Label=alto) e de cada Flag_Eixo_*
               contra o consenso humano, com IC de Wilson
             * Matriz de confusao humano x triangulado (incluindo 'indeterminado' como abstencao)
             * Precisao da ponte por Metodo_Ponte (Confere_Manual S/N) com IC de Wilson

Escala de anotacao (por eixo e geral): 1 = presente | 0 = ausente | ? = nao da para saber pela sinopse/tags
Saidas da avaliacao:
  outputs/reports/gold_standard_avaliacao.txt
  outputs/tables/gold_kappa.csv, gold_metricas_rotulo.csv, gold_confusao.csv, gold_precisao_ponte.csv

Uso:
  python scripts/pipeline_ptbr/gold_standard.py gerar   [--n-por-metodo 20]
  python scripts/pipeline_ptbr/gold_standard.py avaliar
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_ptbr import config as C                                 # noqa: E402
from pipeline_ptbr.dataset import deduplicar_por_obra                 # noqa: E402
from pipeline_ptbr.lexicon import ORDEM_EIXOS, COLUNA_EIXO, EIXOS     # noqa: E402
from pipeline_ptbr.eda_stats import wilson_ic                         # noqa: E402

GOLD = Path(C.GOLD_DIR)
CEGA = GOLD / "anotacao_eixos_cega.csv"
CHAVE = GOLD / "anotacao_eixos_chave.csv"
PONTE_AMOSTRA = GOLD / "auditoria_ponte_amostra.csv"
PROTOCOLO = GOLD / "PROTOCOLO_ANOTACAO.md"
CEGA_ANTIGA = GOLD / "anotacao_gold_cega.csv"

EIXO_CURTO = {"sexo": "Sexo", "violencia": "Violencia", "drogas_alcool": "Drogas_Alcool", "abuso_gatilhos": "Abuso_Gatilhos"}
CAMPOS = [EIXO_CURTO[e] for e in ORDEM_EIXOS] + ["Geral"]


# ---------------------------------------------------------------------------
# Fase 1: gerar planilhas
# ---------------------------------------------------------------------------
def _selecionar_obras(ob: pd.DataFrame) -> pd.DataFrame:
    """Reaproveita os ISBNs da planilha antiga (96 obras estratificadas por quartil), se existir;
    senao amostra 24 por quartil de popularidade (seed fixo)."""
    if CEGA_ANTIGA.exists():
        antiga = pd.read_csv(CEGA_ANTIGA, dtype={"ISBN_PT": str})
        sel = ob[ob["ISBN_PT"].astype(str).isin(antiga["ISBN_PT"].astype(str))]
        if len(sel) >= 50:
            return sel
    u = pd.to_numeric(ob["Qtd_Usuarios_Hardcover"], errors="coerce")
    q = pd.qcut(u.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    return pd.concat([g.sample(min(24, len(g)), random_state=42) for _, g in ob.groupby(q, observed=True)])


def gerar(n_por_metodo: int = 20) -> None:
    C.garantir_dirs()
    GOLD.mkdir(parents=True, exist_ok=True)
    rot = pd.read_csv(C.OUT_LIVROS_ROTULADO, dtype={"ISBN_PT": str})
    ob = deduplicar_por_obra(rot).reset_index(drop=True)
    sel = _selecionar_obras(ob).copy()
    u = pd.to_numeric(ob["Qtd_Usuarios_Hardcover"], errors="coerce")
    quart = pd.qcut(u.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    sel["Q"] = quart.loc[sel.index].astype(str)
    sel = sel.sample(frac=1.0, random_state=7)          # embaralha para o anotador nao ver blocos por quartil

    # Planilha CEGA: contexto para julgar + colunas de resposta. SEM rotulo, SEM V_*, SEM Flag_*.
    ctx_cols = ["ISBN_PT", "Titulo_PT", "Autor_PT", "Editora_PT", "Sinopse_PT",
                "Generos_Hardcover", "Moods_Hardcover", "Tags_Gerais_Hardcover", "Content_Warnings", "Categorias_GB"]
    cega = sel[[c for c in ctx_cols if c in sel.columns]].copy()
    for an in ("A", "B"):
        for campo in CAMPOS:
            cega[f"{an}_{campo}"] = ""
        cega[f"{an}_Obs"] = ""
    if not CEGA.exists():
        cega.to_csv(CEGA, index=False, encoding="utf-8")
        print(f"[gold] planilha cega criada: {CEGA} ({len(cega)} obras)")
    else:
        print(f"[gold] planilha cega ja existe (nao sobrescrita): {CEGA}")

    # CHAVE: o que o pipeline decidiu (para comparar depois). Nao abrir antes de anotar.
    chave_cols = ["ISBN_PT", "Titulo_PT", "Q", "Sensivel_Label", "Sensivel_Label_Confianca", "Sensivel_Label_Fontes",
                  "Sensivel_Label_sem_sinopse", "V_tags", "V_vocab", "V_sinopse", "V_maturity", "V_cw",
                  "Via_Termo_Conteudo", "Spice_Nivel", "Spice_Status"] + [COLUNA_EIXO[e] for e in ORDEM_EIXOS]
    sel[[c for c in chave_cols if c in sel.columns]].to_csv(CHAVE, index=False, encoding="utf-8")
    print(f"[gold] gabarito salvo: {CHAVE}")

    # Amostra da auditoria da ponte: n por metodo (todas se houver menos), para preencher Confere_Manual
    if os.path.exists(C.AUDITORIA_CSV):
        aud = pd.read_csv(C.AUDITORIA_CSV, dtype={"ISBN_PT": str})
        if not PONTE_AMOSTRA.exists():
            partes = [g.sample(min(n_por_metodo, len(g)), random_state=42)
                      for _, g in aud.groupby("Metodo_Ponte")]
            am = pd.concat(partes) if partes else aud.iloc[0:0]
            # garante que todos os Autor_Confere == False entrem (sao os casos mais suspeitos)
            sus = aud[aud["Autor_Confere"].astype(str).str.lower() == "false"]
            am = pd.concat([am, sus]).drop_duplicates("ISBN_PT")
            am["Confere_Manual"] = ""
            am["Obs"] = ""
            am.to_csv(PONTE_AMOSTRA, index=False, encoding="utf-8")
            print(f"[gold] amostra da ponte criada: {PONTE_AMOSTRA} ({len(am)} edicoes; "
                  f"por metodo: {am['Metodo_Ponte'].value_counts().to_dict()})")
        else:
            print(f"[gold] amostra da ponte ja existe (nao sobrescrita): {PONTE_AMOSTRA}")

    _escrever_protocolo()


def _escrever_protocolo() -> None:
    linhas = [
        "# Protocolo de Anotação — Gold Standard (conteúdo sensível, 4 eixos)",
        "",
        "## Arquivos",
        f"- `{CEGA.name}`: preencher. Uma linha por obra. **Não consulte** `{CHAVE.name}` antes de terminar.",
        f"- `{PONTE_AMOSTRA.name}`: preencher `Confere_Manual` com `S` (a obra do Hardcover é a mesma da edição PT-BR) ou `N`.",
        "",
        "## Quem anota",
        "Dois anotadores independentes (A e B). Cada um preenche **apenas** as suas colunas (`A_*` ou `B_*`), sem ver as do outro.",
        "Sugestão prática: duplique o arquivo, cada anotador preenche a sua cópia, depois junte as colunas.",
        "",
        "## O que ler para decidir",
        "Use SOMENTE o que está na linha: `Sinopse_PT`, `Generos_Hardcover`, `Moods_Hardcover`, `Tags_Gerais_Hardcover`,",
        "`Content_Warnings`, `Categorias_GB`. Pode usar conhecimento prévio da obra se tiver lido; anote isso em `*_Obs`.",
        "",
        "## Escala (igual para cada eixo e para o Geral)",
        "| Valor | Significado |",
        "| :---: | :--- |",
        "| `1` | o tema está presente de forma relevante na obra |",
        "| `0` | há evidência de que o tema está ausente / a obra é leve |",
        "| `?` | não dá para saber com o material disponível (abstenção) |",
        "",
        "## Definição operacional de cada eixo",
    ]
    for e in ORDEM_EIXOS:
        ex_en = ", ".join(t for t in EIXOS[e]["cw"][:6])
        ex_pt = ", ".join(t for t in EIXOS[e]["pt"][:5])
        linhas += [f"### {EIXO_CURTO[e]} — {EIXOS[e]['rotulo']}",
                   f"- Marque `1` se houver cena, tema central ou alerta explícito. Exemplos de sinais (EN): {ex_en}. (PT): {ex_pt}.",
                   "- Menção de passagem (ex.: 'tomou uma taça de vinho') **não** conta para `1` em Drogas/Álcool; alcoolismo, embriaguez recorrente ou uso de drogas contam.",
                   "- Violência sexual (estupro, assédio) marca `1` em **Sexo** E em **Abuso_Gatilhos**.",
                   ""]
    linhas += [
        "### Geral — a obra tem conteúdo sensível/problemático (qualquer eixo) a ponto de ser inadequada para leitores < 16 anos?",
        "- `1` se qualquer eixo for `1` com peso relevante; `0` se a obra for claramente leve (YA/romcom clean); `?` caso contrário.",
        "",
        "## Decisões de desempate (anote em Obs)",
        "- `dark romance` na sinopse/tag ⇒ pelo menos Sexo=`1` ou `?`, nunca `0`.",
        "- `Young Adult` sozinho **não** implica `0`: YA pode ter violência/abuso. Decida pelo conteúdo.",
        "- Se a sinopse for só marketing (sem enredo), prefira `?`.",
        "",
        "## Depois de preencher",
        "```bash",
        "python scripts/pipeline_ptbr/gold_standard.py avaliar",
        "```",
        "Gera κ entre anotadores, precisão/recall do rótulo automático por eixo e precisão da ponte por método.",
    ]
    PROTOCOLO.write_text("\n".join(linhas), encoding="utf-8")
    print(f"[gold] protocolo salvo: {PROTOCOLO}")


# ---------------------------------------------------------------------------
# Fase 2: avaliar
# ---------------------------------------------------------------------------
def _norm_resp(s: pd.Series) -> pd.Series:
    """'1'/'0'/'?' -> 1.0 / 0.0 / NaN (aceita 'sim/nao', 's/n', 'x')."""
    m = {"1": 1.0, "0": 0.0, "s": 1.0, "n": 0.0, "sim": 1.0, "nao": 0.0, "não": 0.0, "x": 1.0, "?": np.nan, "": np.nan}
    return s.astype(str).str.strip().str.lower().map(lambda x: m.get(x, np.nan))


def cohen_kappa(a: pd.Series, b: pd.Series) -> Tuple[float, float, int]:
    m = pd.concat([a, b], axis=1).dropna()
    n = len(m)
    if n == 0:
        return (np.nan, np.nan, 0)
    x, y = m.iloc[:, 0], m.iloc[:, 1]
    po = float((x == y).mean())
    pe = sum((x == c).mean() * (y == c).mean() for c in set(x) | set(y))
    k = (po - pe) / (1 - pe) if pe < 1 else (1.0 if po == 1 else 0.0)
    return (k, po, n)


def _metricas_binarias(pred: pd.Series, gold: pd.Series) -> Dict[str, float]:
    """pred/gold booleanos alinhados (gold sem NaN). Precisao, recall, F1 com IC Wilson para P e R."""
    m = pd.concat([pred.astype(bool), gold], axis=1).dropna()
    p_, g_ = m.iloc[:, 0].astype(bool), m.iloc[:, 1].astype(bool)
    tp = int((p_ & g_).sum()); fp = int((p_ & ~g_).sum()); fn = int((~p_ & g_).sum()); tn = int((~p_ & ~g_).sum())
    prec, plo, phi = wilson_ic(tp, tp + fp)
    rec, rlo, rhi = wilson_ic(tp, tp + fn)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) and not np.isnan(prec) and not np.isnan(rec) else np.nan
    return {"n": len(m), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precisao": prec, "prec_IC_inf": plo, "prec_IC_sup": phi,
            "recall": rec, "rec_IC_inf": rlo, "rec_IC_sup": rhi, "F1": f1}


def avaliar() -> str:
    C.garantir_dirs()
    T, R = Path(C.OUT_TABLES), Path(C.OUT_REPORTS)
    L: List[str] = []
    w = L.append
    w("AVALIAÇÃO DO GOLD STANDARD\n" + "=" * 90)

    # ---- 1. rotulo
    if CEGA.exists() and CHAVE.exists():
        cega = pd.read_csv(CEGA, dtype=str).fillna("")
        chave = pd.read_csv(CHAVE, dtype={"ISBN_PT": str})
        d = cega.merge(chave, on="ISBN_PT", suffixes=("", "_chave"))
        respA = {c: _norm_resp(d[f"A_{c}"]) for c in CAMPOS}
        respB = {c: _norm_resp(d[f"B_{c}"]) for c in CAMPOS}
        n_pre = {c: int((respA[c].notna() | respB[c].notna()).sum()) for c in CAMPOS}
        if max(n_pre.values()) == 0:
            w(f"\n[rótulo] Planilha {CEGA.name} ainda sem respostas. Preencha as colunas A_*/B_* e rode de novo.")
        else:
            kap = []
            for c in CAMPOS:
                k, po, n = cohen_kappa(respA[c], respB[c])
                kap.append({"Campo": c, "n_pares_anotados": n, "concordancia_bruta": round(po, 3) if n else np.nan,
                            "kappa_cohen": round(k, 3) if n else np.nan})
            kap = pd.DataFrame(kap)
            kap.to_csv(T / "gold_kappa.csv", index=False, encoding="utf-8")
            w("\n1) CONCORDÂNCIA ENTRE ANOTADORES (Cohen's κ; '?' é tratado como ausência de resposta)\n" + kap.to_string(index=False))
            w("   Referência (Landis & Koch): <0,20 fraca | 0,21–0,40 razoável | 0,41–0,60 moderada | 0,61–0,80 substancial | >0,80 quase perfeita")

            # consenso humano: media das respostas disponiveis; 1 se media >= 0.5; descarta se ambos '?'
            cons = {}
            for c in CAMPOS:
                m = pd.concat([respA[c], respB[c]], axis=1).mean(axis=1)
                cons[c] = m.map(lambda x: np.nan if pd.isna(x) else bool(x >= 0.5))
            met = []
            # Sensivel_Label=alto vs Geral
            pred_alto = d["Sensivel_Label"] == "alto"
            r = _metricas_binarias(pred_alto, cons["Geral"]); r["Predito"] = "Sensivel_Label == alto"; r["Gold"] = "Geral"
            met.append(r)
            pred_alto_ns = d.get("Sensivel_Label_sem_sinopse", pd.Series("", index=d.index)) == "alto"
            r = _metricas_binarias(pred_alto_ns, cons["Geral"]); r["Predito"] = "Sensivel_Label_sem_sinopse == alto"; r["Gold"] = "Geral"
            met.append(r)
            for e in ORDEM_EIXOS:
                col = COLUNA_EIXO[e]
                if col in d.columns:
                    pred = d[col].astype(str).str.lower().isin(["true", "1"])
                    r = _metricas_binarias(pred, cons[EIXO_CURTO[e]]); r["Predito"] = col; r["Gold"] = EIXO_CURTO[e]
                    met.append(r)
            met = pd.DataFrame(met)[["Predito", "Gold", "n", "TP", "FP", "FN", "TN", "precisao", "prec_IC_inf", "prec_IC_sup",
                                     "recall", "rec_IC_inf", "rec_IC_sup", "F1"]]
            met.to_csv(T / "gold_metricas_rotulo.csv", index=False, encoding="utf-8")
            w("\n2) RÓTULO AUTOMÁTICO × CONSENSO HUMANO (obras em que ao menos um anotador respondeu 1/0)\n" + met.round(3).to_string(index=False))

            # matriz de confusao 3x3 (humano Geral: 1/0/?  x  Sensivel_Label)
            hum = pd.concat([respA["Geral"], respB["Geral"]], axis=1).mean(axis=1)
            hum_cat = hum.map(lambda x: "?" if pd.isna(x) else ("1 (sensível)" if x >= 0.5 else "0 (leve)"))
            conf = pd.crosstab(hum_cat.rename("Humano_Geral"), d["Sensivel_Label"].rename("Sensivel_Label"))
            conf.to_csv(T / "gold_confusao.csv", encoding="utf-8")
            w("\n3) MATRIZ HUMANO × TRIANGULADO ('indeterminado' = abstenção do pipeline; '?' = abstenção humana)\n" + conf.to_string())
            w("   Leitura: células (0 (leve), alto) = falsos positivos do pipeline; (1 (sensível), indeterminado) = recall perdido por abstenção.")
    else:
        w(f"\n[rótulo] Rode primeiro: python scripts/pipeline_ptbr/gold_standard.py gerar")

    # ---- 2. ponte
    fontes_ponte = [p for p in (PONTE_AMOSTRA, Path(C.AUDITORIA_CSV)) if p.exists()]
    if fontes_ponte:
        aud = pd.concat([pd.read_csv(p, dtype=str) for p in fontes_ponte]).drop_duplicates("ISBN_PT", keep="first")
        resp = _norm_resp(aud["Confere_Manual"]) if "Confere_Manual" in aud.columns else pd.Series(np.nan, index=aud.index)
        aud = aud.assign(_ok=resp)
        if aud["_ok"].notna().sum() == 0:
            w(f"\n[ponte] Nenhum Confere_Manual preenchido ainda em {PONTE_AMOSTRA.name}. Preencha S/N e rode de novo.")
        else:
            rows = []
            for m, g in aud.groupby("Metodo_Ponte"):
                gg = g.dropna(subset=["_ok"])
                k, n = int(gg["_ok"].sum()), len(gg)
                p, lo, hi = wilson_ic(k, n)
                rows.append({"Metodo_Ponte": m, "n_auditadas": n, "n_corretas": k, "precisao": round(p, 3) if n else np.nan,
                             "IC95_inf": round(lo, 3) if n else np.nan, "IC95_sup": round(hi, 3) if n else np.nan,
                             "n_total_no_metodo": len(g)})
            gg = aud.dropna(subset=["_ok"])
            k, n = int(gg["_ok"].sum()), len(gg)
            p, lo, hi = wilson_ic(k, n)
            rows.append({"Metodo_Ponte": "TODOS (auditados)", "n_auditadas": n, "n_corretas": k, "precisao": round(p, 3),
                         "IC95_inf": round(lo, 3), "IC95_sup": round(hi, 3), "n_total_no_metodo": len(aud)})
            pp = pd.DataFrame(rows)
            pp.to_csv(T / "gold_precisao_ponte.csv", index=False, encoding="utf-8")
            w("\n4) PRECISÃO DA PONTE POR MÉTODO (Confere_Manual; IC de Wilson 95%)\n" + pp.to_string(index=False))
            w("   Nota: matches por ISBN (score 1.0) entram na auditoria só quando Autor_Confere=False; "
              "a precisão dos demais métodos é a métrica que legitima os limiares FUZZY_ACEITE/EMB_ACEITE/EMB_MARGEM.")
    txt = "\n".join(L)
    (R / "gold_standard_avaliacao.txt").write_text(txt, encoding="utf-8")
    print(txt)
    return txt


def main() -> None:
    ap = argparse.ArgumentParser(description="Gold standard: gerar planilhas cegas / avaliar anotações")
    ap.add_argument("acao", choices=["gerar", "avaliar"])
    ap.add_argument("--n-por-metodo", type=int, default=20, help="amostra da auditoria da ponte por Metodo_Ponte")
    a = ap.parse_args()
    gerar(a.n_por_metodo) if a.acao == "gerar" else avaliar()


if __name__ == "__main__":
    main()
