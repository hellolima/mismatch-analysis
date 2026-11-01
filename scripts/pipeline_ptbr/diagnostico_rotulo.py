"""Diagnóstico do rótulo triangulado de conteúdo sensível (rodar DEPOIS de rotulagem.py).

Responde, com os seus dados:
  1. De onde vem cada 'alto' (combinações de fontes) e cada 'baixo'/'provavel_baixo'
  2. Circularidade: o voto de sinopse 'alto' concentra-se nas obras que entraram no seed via 'termo:conteudo'?
     + distribuição do rótulo SEM essas obras
  3. Recall relativo das fontes tomando as tags do Hardcover como pseudo-referência (alta precisão, n pequeno)
  4. Vocabulário de conteúdo sensível (sexo/spice, drogas/álcool, violência, abuso/gatilhos) e distribuição por eixo temático
  5. Lista de revisão manual: obras 'baixo'/'provavel_baixo' em estratos que esperamos com conteúdo sensível

Saídas:
  outputs/reports/diagnostico_rotulo.txt
  outputs/audits/diagnostico_baixo_revisar.csv
  outputs/audits/diagnostico_peppers.csv

Uso: python scripts/pipeline_ptbr/diagnostico_rotulo.py [--ano-min 2015] [--entrada CSV]
"""
import os
import re
import sys
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_ptbr import config as C                             # noqa: E402
from pipeline_ptbr.dataset import (                               # noqa: E402
    filtrar_escopo,
    deduplicar_por_obra,
    resolver_colunas_rotulo,
)
from pipeline_ptbr.rotulagem import eixos_sensiveis_obra          # noqa: E402

RE_PEPPER = re.compile(r"(\d)\s*peppers?")
RE_SENSIVEL_LIVRE = re.compile(
    r"spic|steam|open door|smut|explicit|heat|erotic|pepper|sex|hot"
    r"|violen|murder|gore|tortur|blood|kidnap|kill|assault|gun"
    r"|drug|alcohol|drunk|drink|addict|substance|overdose|narcotic"
    r"|abuse|rape|toxic|suicid|self.?harm|trauma|stalk"
)
# Alias retrocompativel
RE_SPICE_LIVRE = RE_SENSIVEL_LIVRE
ESTRATOS_ESPERA_SENSIVEL = ("new_adult", "dark", "romcom", "esportivo", "romantasia")
ESTRATOS_ESPERA_SPICE = ESTRATOS_ESPERA_SENSIVEL


def tags_de(s) -> list:
    return [t.strip().lower() for t in str(s).split("|") if t.strip() and str(s) != "nan"]


def executar_diagnostico_rotulo(entrada: Path = Path(C.OUT_LIVROS_ROTULADO),
                                ano_min: int = C.ANO_MIN_PADRAO) -> str:
    C.garantir_dirs()
    df_bruto = pd.read_csv(entrada)
    df_filtrado, _ = filtrar_escopo(df_bruto, ano_min)
    ob = deduplicar_por_obra(df_filtrado).reset_index(drop=True)

    col_label, _, col_fontes = resolver_colunas_rotulo(ob)

    # Garante colunas de eixos sensiveis se nao estiverem presentes no CSV
    cols_eixos = ("Flag_Eixo_Sexo", "Flag_Eixo_Violencia", "Flag_Eixo_Drogas_Alcool", "Flag_Eixo_Abuso_Gatilhos")
    if not all(c in ob.columns for c in cols_eixos):
        eixos_df = ob.apply(eixos_sensiveis_obra, axis=1, result_type="expand")
        for c in cols_eixos:
            ob[c] = eixos_df[c]

    L = []
    w = L.append
    w(f"DIAGNÓSTICO DO RÓTULO ({col_label}) | obras: {len(ob)} (ano >= {ano_min}, sem formatos excluídos)\n" + "=" * 90)

    # ---- 1. composição
    w("\n1) COMPOSIÇÃO DO RÓTULO POR COMBINAÇÃO DE FONTES")
    comp = ob.groupby([col_label, ob[col_fontes].fillna("")]).size()
    w(comp.to_string())
    nb = int(ob[col_label].isin(["baixo", "provavel_baixo"]).sum())
    so_ya = int(((ob[col_label].isin(["baixo", "provavel_baixo"])) & (ob[col_fontes] == "cobertura_ya")).sum())
    w(f"\n  'baixo'/'provavel_baixo' sem NENHUM voto próprio (só regra cobertura+YA): {so_ya}/{nb}")
    w("  Se for ~100%, 'provavel_baixo' é uma INFERÊNCIA a partir do gênero YA, não uma medição: não use como negativo "
      "sem validação humana (e evite o atalho 'capa fofa/YA => baixo' no POC II).")

    # ---- 1b. distribuicao por eixo tematico de conteudo sensivel
    w("\n1b) PREVALÊNCIA POR EIXO DE CONTEÚDO SENSÍVEL (nível obra)")
    nomes_eixos = {
        "Flag_Eixo_Sexo": "Sexo / Erótico / Spice",
        "Flag_Eixo_Violencia": "Violência / Crime / Gore",
        "Flag_Eixo_Drogas_Alcool": "Drogas / Bebidas Alcoólicas",
        "Flag_Eixo_Abuso_Gatilhos": "Abuso / Violência Sexual / Saúde Mental",
    }
    for col_eixo, desc_eixo in nomes_eixos.items():
        if col_eixo in ob.columns:
            k = int(ob[col_eixo].astype(bool).sum())
            w(f"  {col_eixo:<26} ({desc_eixo:<39}): {k:>3}/{len(ob)} ({100 * k / max(len(ob), 1):.1f}%)")

    # ---- 2. circularidade com o canal de seed
    w("\n2) CIRCULARIDADE COM O CANAL 'termo:conteudo' (seed buscou por termos de conteúdo adulto)")
    via = ob["Tipos_Origem"].fillna("").str.contains("termo:conteudo") if "Tipos_Origem" in ob.columns else pd.Series(False, index=ob.index)
    w(f"  Obras que entraram via termo:conteudo: {int(via.sum())}/{len(ob)}")
    w("  V_sinopse x via_termo_conteudo:")
    w("    " + pd.crosstab(ob["V_sinopse"].fillna("abstem"), via.rename("via_conteudo")).to_string().replace("\n", "\n    "))
    w(f"  {col_label} x via_termo_conteudo:")
    w("    " + pd.crosstab(ob[col_label], via.rename("via_conteudo")).to_string().replace("\n", "\n    "))
    w(f"  {col_label} SEM as obras via termo:conteudo (análise de sensibilidade):")
    w("    " + ob.loc[~via, col_label].value_counts().to_string().replace("\n", "\n    "))

    # ---- 3. recall relativo (tags como pseudo-referência)
    w("\n3) RECALL RELATIVO: entre obras com V_tags=alto, % em que cada outra fonte também vota alto")
    ref = ob[ob["V_tags"] == "alto"]
    w(f"  n (referência, V_tags=alto) = {len(ref)}  [n pequeno: leia como ordem de grandeza]")
    for f in ("V_vocab", "V_sinopse", "V_maturity", "V_cw"):
        if f in ref.columns:
            k = int((ref[f] == "alto").sum())
            w(f"  {f:<11} {k}/{len(ref)} = {100 * k / max(len(ref), 1):.0f}%")
    w("  Recall baixo da sinopse/maturity = o léxico/fonte perde muita coisa; o 'indeterminado' esconde conteúdo sensível.")

    # ---- 4. vocabulário de conteúdo sensível
    w("\n4) TAGS COM VOCABULÁRIO DE CONTEÚDO SENSÍVEL (sexo, violência, drogas/álcool, abuso) em Tags_Gerais/Generos/Moods/CW")
    cont = Counter()
    for col in ("Tags_Gerais_Hardcover", "Generos_Hardcover", "Moods_Hardcover", "Content_Warnings"):
        if col in ob.columns:
            for s in ob[col]:
                for t in set(tags_de(s)):
                    if RE_SENSIVEL_LIVRE.search(t):
                        cont[t] += 1
    w("  " + (", ".join(f"{t} ({n})" for t, n in cont.most_common(45)) or "(nenhuma)"))
    linhas = []
    for _, r in ob.iterrows():
        for t in tags_de(r.get("Tags_Gerais_Hardcover")):
            m = RE_PEPPER.search(t)
            if m:
                linhas.append({
                    "Titulo_PT": r.get("Titulo_PT"),
                    "Autor_PT": r.get("Autor_PT"),
                    "Tag": t,
                    "Peppers": int(m.group(1)),
                    "V_tags": r.get("V_tags"),
                    col_label: r.get(col_label),
                })
    pep = pd.DataFrame(linhas)
    pep.to_csv(os.path.join(C.OUT_AUDITS, "diagnostico_peppers.csv"), index=False, encoding="utf-8")
    w(f"\n  Obras com tag 'N peppers': {len(pep)} (outputs/audits/diagnostico_peppers.csv). Cruze com {col_label}: "
      f"se 4-5 peppers aparecem como 'indeterminado' ou 'baixo', o labeling.py não está lendo essa escala.")
    if len(pep):
        w("    " + pd.crosstab(pep["Peppers"], pep[col_label]).to_string().replace("\n", "\n    "))

    # ---- 5. revisão manual dos 'baixo' / 'provavel_baixo' suspeitos
    w("\n5) 'BAIXO' / 'PROVAVEL_BAIXO' EM ESTRATOS QUE ESPERAMOS COM CONTEÚDO SENSÍVEL (revisar à mão)")
    est = ob["Estratos_Origem"].fillna("") if "Estratos_Origem" in ob.columns else pd.Series("", index=ob.index)
    sus = ob[ob[col_label].isin(["baixo", "provavel_baixo"]) & est.apply(lambda s: any(e in s for e in ESTRATOS_ESPERA_SENSIVEL))]
    cols = [c for c in ("Titulo_PT", "Autor_PT", "Estratos_Origem", "Generos_Hardcover", "Categorias_GB",
                        "Maturity_GB", "Qtd_Usuarios_Hardcover", col_label, col_fontes) if c in sus.columns]
    sus[cols].to_csv(os.path.join(C.OUT_AUDITS, "diagnostico_baixo_revisar.csv"), index=False, encoding="utf-8")
    w(f"  {len(sus)} obras em outputs/audits/diagnostico_baixo_revisar.csv. Cada falso 'baixo' aqui é evidência de que a regra YA erra.")

    txt = "\n".join(L)
    Path(os.path.join(C.OUT_REPORTS, "diagnostico_rotulo.txt")).write_text(txt, encoding="utf-8")
    print(txt)
    return txt


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnóstico do rótulo triangulado (v3.1)")
    ap.add_argument("--entrada", type=Path, default=Path(C.OUT_LIVROS_ROTULADO))
    ap.add_argument("--ano-min", type=int, default=C.ANO_MIN_PADRAO)
    a = ap.parse_args()
    executar_diagnostico_rotulo(a.entrada, a.ano_min)


if __name__ == "__main__":
    main()