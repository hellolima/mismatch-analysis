"""Rotulo de CONTEUDO INADEQUADO / SENSIVEL por triangulacao de fontes, com abstencao explicita (v3.1).

Escopo Ampliado do Projeto:
  O rotulo `Sensivel_Label` nao mede apenas 'spice' (erotismo), mas CONTEUDOS SENSIVEIS E PROBLEMATICOS
  NO GERAL nos 4 eixos:
    1. Conteudo Sexual / Erotico (spice, smut, cenas explicitas, violencia/assedio sexual)
    2. Drogas e Bebidas Alcoolicas (alcoolismo, embriaguez, uso/trafico de drogas, vicio, overdose)
    3. Violencia e Brutalidade (violencia fisica/grafica, assassinato, tortura, sequestro, gore, armas)
    4. Abuso, Toxicidade e Saude Mental (abuso domestico/infantil/psicologico, automutilacao, suicidio, stalking)

Principio: cada fonte de evidencia vota 'alto', 'baixo' ou se ABSTEM (None). Ausencia de evidencia
NUNCA vira 'baixo' por padrao (a ausencia de tag e dado faltante nao aleatorio: depende de popularidade).

FONTES (por edicao; agregadas por OBRA; votos divergentes entre edicoes = 'conflito'):
  V_tags      Hardcover: Spice_Nivel >= 2 -> alto; 0 ou Closed_Door -> baixo; 1 -> abstem
  V_vocab     vocabulario lido em Tags_Gerais/Generos: escala "N peppers"/"spice level N"
              (N >= PEPPER_ALTO -> alto; N <= PEPPER_BAIXO -> baixo) + termos de conteudo explicito,
              drogas/alcool, violencia e temas sensiveis -> alto; closed door, clean/sweet romance,
              low heat, family friendly, all ages -> baixo
  V_sinopse   lexico PT na Sinopse_PT: termos explicitos, drogas/bebidas/alcool, violencia e abuso -> alto;
              'clean/porta fechada', 'para todas as idades' -> baixo
  V_maturity  Maturity_GB == MATURE -> alto (senao abstem)
  V_cw        Flag_CW_Sexual_Content / Flag_Has_CW / Content_Warnings sensiveis (sexo, violencia,
              drogas/alcool, abuso/suicidio) -> alto (FORTE, pois indica conteudo sensivel/inadequado)

FAMILIAS INDEPENDENTES (para 'forte'): comunidade (= tags + vocab, mesma origem: leitores do Hardcover),
sinopse, maturity. CW conta como validador direto para rotulo alto.

CIRCULARIDADE: obras que entraram no seed via 'termo:conteudo' (Via_Termo_Conteudo=True) casam com o lexico
da sinopse POR CONSTRUCAO (a busca no Google Books usou esse termo). O voto de sinopse delas continua
registrado, mas NAO conta como familia independente e, sozinho, NAO sustenta rotulo (-> circular).

Saidas:
  data/processed/livros_ptbr_v3_rotulado.csv
  outputs/reports/rotulagem_resumo.txt
  outputs/audits/rotulagem_conflitos.csv

Uso: python scripts/pipeline_ptbr/rotulagem.py [--entrada CSV] [--ano-min 2015] [--pop-pct 50]
"""
import os
import re
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_ptbr import config as C                             # noqa: E402
from pipeline_ptbr.common import sem_acento                       # noqa: E402
from pipeline_ptbr.util import como_bool                          # noqa: E402
from pipeline_ptbr.dataset import filtrar_escopo, chave_obra      # noqa: E402
from pipeline_ptbr.lexicon import (                               # noqa: E402  (fonte unica do lexico)
    RE_ALTO, RE_BAIXO, RE_VOC_ALTO, RE_VOC_BAIXO, RE_CW_SENSIVEL, RE_POR_EIXO, ORDEM_EIXOS, COLUNA_EIXO,
)

FONTES = ("tags", "vocab", "sinopse", "maturity", "cw")
PEPPER_ALTO = 3    # escala 0-5: >=3 'alto'
PEPPER_BAIXO = 1   # <=1 'baixo'; 2 = abstem (ambiguo)

# Retrocompatibilidade: nomes antigos exportados por este modulo
RE_CW_SEXUAL = RE_POR_EIXO["sexo"]["cw"]
RE_CW_VIOLENCIA = RE_POR_EIXO["violencia"]["cw"]
RE_CW_DROGAS_ALCOOL = RE_POR_EIXO["drogas_alcool"]["cw"]
RE_CW_ABUSO_GATILHOS = RE_POR_EIXO["abuso_gatilhos"]["cw"]


def _v(x: object) -> Optional[str]:
    """NaN/None -> None (abstencao); string -> string."""
    return x if isinstance(x, str) else None


RE_PEPPER = re.compile(r"(?<!\d)(\d)\s*peppers?\b|\bspice[\s\-]*(?:level)?[\s\-]*(\d)\b")

# YA / infantil: casa a TAG inteira (nao substring de texto livre)
RE_YA_TAG = re.compile(
    r"^(?:young adult\b.*|middle grade\b.*|juvenile\b.*|children'?s?(?: books?| fiction| literature)?|"
    r"infantil\b.*|infantojuvenil\b.*)$"
)


def voto_tags(r: pd.Series) -> Optional[str]:
    nivel = pd.to_numeric(r.get("Spice_Nivel"), errors="coerce")
    if pd.notna(nivel):
        if nivel >= 2:
            return "alto"
        if nivel == 0:
            return "baixo"
        # nivel == 1 (sensual/implicito, ex.: 'sexy'): nao e evidencia de conteudo BAIXO -> abstem e cai nas flags
    if como_bool(r.get("Flag_Spice_Tag")):
        return "alto"
    if como_bool(r.get("Flag_Closed_Door_Tag")):
        return "baixo"
    return None


def voto_vocab(r: pd.Series) -> Optional[str]:
    tags = []
    for col in ("Tags_Gerais_Hardcover", "Generos_Hardcover"):
        s = r.get(col)
        if isinstance(s, str):
            tags += [sem_acento(t).strip() for t in s.split("|") if t.strip()]
    altos = baixos = 0
    for t in tags:
        m = RE_PEPPER.search(t)
        if m:
            n = int(m.group(1) or m.group(2))
            if n >= PEPPER_ALTO:
                altos += 1
            elif n <= PEPPER_BAIXO:
                baixos += 1
            continue
        if RE_VOC_BAIXO.search(t):
            baixos += 1
        elif RE_VOC_ALTO.search(t):
            altos += 1
    if altos and baixos:
        return "conflito"
    return "alto" if altos else ("baixo" if baixos else None)


def voto_sinopse(r: pd.Series) -> Optional[str]:
    t = sem_acento(r.get("Sinopse_PT"))
    if not t:
        return None
    a, b = bool(RE_ALTO.search(t)), bool(RE_BAIXO.search(t))
    if a and b:
        return "conflito"
    return "alto" if a else ("baixo" if b else None)


def voto_maturity(r: pd.Series) -> Optional[str]:
    val = r.get("Maturity_GB")
    if pd.isna(val) or val is None:
        return None
    return "alto" if str(val).strip().upper() == "MATURE" else None


def voto_cw(r: pd.Series) -> Optional[str]:
    """Avalia flags de gatilho/sensiveis existentes E varre Content_Warnings para os 4 eixos
    (sexo, violencia, drogas/alcool, abuso/saude mental).
    """
    for flag_col in (
        "Flag_CW_Sexual_Content",
        "Flag_CW_Violencia",
        "Flag_CW_Drogas_Alcool",
        "Flag_CW_Abuso_Gatilhos",
        "Flag_Has_CW",
    ):
        if flag_col in r and como_bool(r.get(flag_col)):
            return "alto"
    cws = sem_acento(r.get("Content_Warnings"))
    if cws and RE_CW_SENSIVEL.search(cws):
        return "alto"
    return None


def eixos_sensiveis_obra(r: pd.Series) -> dict:
    """Identifica quais dos 4 eixos de conteudo sensivel/problematico estao presentes na obra
    (Flags de CW do labeling + Content_Warnings/Tags/Generos em EN + Sinopse_PT), usando `lexicon.EIXOS`.
    """
    cws = sem_acento(r.get("Content_Warnings"))
    tags = sem_acento(r.get("Tags_Gerais_Hardcover")) + " | " + sem_acento(r.get("Generos_Hardcover"))
    sin = sem_acento(r.get("Sinopse_PT"))
    texto_en = f"{cws} | {tags}"
    flag_cw = {
        "sexo": "Flag_CW_Sexual_Content",
        "violencia": "Flag_CW_Violencia",
        "drogas_alcool": "Flag_CW_Drogas_Alcool",
        "abuso_gatilhos": "Flag_CW_Abuso_Gatilhos",
    }
    out = {}
    for e in ORDEM_EIXOS:
        rx = RE_POR_EIXO[e]
        hit = (como_bool(r.get(flag_cw[e]))
               or bool(rx["vocab"].search(texto_en))
               or bool(rx["pt"].search(sin)))
        if e == "sexo":
            nivel = pd.to_numeric(r.get("Spice_Nivel"), errors="coerce")
            hit = hit or como_bool(r.get("Flag_Spice_Tag")) or (pd.notna(nivel) and float(nivel) >= 2)
        out[COLUNA_EIXO[e]] = bool(hit)
    return out


def agg_votos(s: pd.Series) -> Optional[str]:
    """Agrega votos das edicoes de uma obra: divergencia = conflito."""
    v = {x for x in s if isinstance(x, str)}
    if not v:
        return None
    if "conflito" in v or {"alto", "baixo"} <= v:
        return "conflito"
    return next(iter(v))


def eh_ya(r: pd.Series) -> bool:
    """YA/infantil por TAG delimitada em Generos_Hardcover ('|') e Categorias_GB ('|', ',', ';')."""
    tags = []
    for col, sep in (("Generos_Hardcover", r"\|"), ("Categorias_GB", r"[|,;]")):
        s = sem_acento(r.get(col))
        if s:
            tags += [t.strip() for t in re.split(sep, s) if t.strip()]
    return any(RE_YA_TAG.match(t) for t in tags)


def _comb(a: Optional[str], b: Optional[str]) -> Optional[str]:
    s = {x for x in (a, b) if x}
    if not s:
        return None
    if "conflito" in s or s == {"alto", "baixo"}:
        return "conflito"
    return next(iter(s))


# ---------------------------------------------------------------------------
# TABELA DE REGRAS DE DECISAO (avaliadas em ordem; a primeira que casar decide).
# Cada regra: (id, descricao, condicao(ctx) -> bool, rotulo, confianca, fontes(ctx) -> str)
# `ctx` e o dicionario produzido por `_contexto_decisao`. Exportavel para LaTeX via `tabela_regras()`.
# ---------------------------------------------------------------------------
REGRAS_DECISAO = [
    ("R1", "Conflito entre familias, ou voto 'baixo' coexistindo com 'alto'/CW",
     lambda c: bool(c["conf_f"]) or (bool(c["baixo_f"]) and (bool(c["alto_f"]) or c["cw"])),
     "indeterminado", "conflito", lambda c: c["com_voto"]),
    ("R2", "Unico voto 'alto' e a sinopse de obra achada via termo:conteudo (circularidade)",
     lambda c: bool(c["alto_f"]) and not c["indep"] and not c["cw"],
     "indeterminado", "circular", lambda c: "sinopse(via_termo_conteudo)"),
    ("R3", ">= 2 familias independentes 'alto', ou Content Warning sensivel",
     lambda c: len(c["indep"]) >= 2 or c["cw"],
     "alto", "forte", lambda c: c["det_alto"]),
    ("R4", "Exatamente 1 familia independente 'alto'",
     lambda c: bool(c["alto_f"]),
     "alto", "fraca", lambda c: c["det_alto"]),
    ("R5", ">= 2 familias 'baixo' (closed door, clean, etc.)",
     lambda c: len(c["baixo_f"]) >= 2,
     "baixo", "forte", lambda c: c["det_baixo"]),
    ("R6", "Exatamente 1 familia 'baixo'",
     lambda c: bool(c["baixo_f"]),
     "baixo", "fraca", lambda c: c["det_baixo"]),
    ("R7", "Nenhum voto, mas obra YA com boa cobertura comunitaria (inferencia, nao medicao)",
     lambda c: c["cobertura_boa"] and c["ya"],
     "provavel_baixo", "nao_validado", lambda c: "cobertura_ya"),
    ("R8", "Nenhum voto e cobertura insuficiente: abstencao honesta",
     lambda c: True,
     "indeterminado", "sem_evidencia", lambda c: ""),
]


def _contexto_decisao(r: pd.Series, usar_sinopse: bool = True, ignorar: tuple = ()) -> dict:
    """Monta o contexto consumido pelas regras. `ignorar` zera fontes (para ablacao)."""
    v = {f: _v(r[f"V_{f}"]) for f in FONTES}
    if not usar_sinopse:
        v["sinopse"] = None
    for f in ignorar:
        v[f] = None
    via = bool(r["Via_Termo_Conteudo"])
    fam = {"comunidade": _comb(v["tags"], v["vocab"]), "sinopse": v["sinopse"], "maturity": v["maturity"]}
    alto_f = [k for k, x in fam.items() if x == "alto"]
    return {
        "alto_f": alto_f,
        "baixo_f": [k for k, x in fam.items() if x == "baixo"],
        "conf_f": [k for k, x in fam.items() if x == "conflito"],
        "cw": v["cw"] == "alto",
        "indep": [f for f in alto_f if not (f == "sinopse" and via)],
        "com_voto": "|".join(f for f in FONTES if v[f] in ("alto", "baixo", "conflito")),
        "det_alto": "|".join(f for f in FONTES if v[f] == "alto"),
        "det_baixo": "|".join(f for f in FONTES if v[f] == "baixo"),
        "cobertura_boa": bool(r["Cobertura_Boa"]),
        "ya": bool(r["YA"]),
    }


def decidir(r: pd.Series, usar_sinopse: bool = True, ignorar: tuple = ()):
    """Aplica REGRAS_DECISAO em ordem. usar_sinopse=False: alvo para o modulo de texto (anti-leakage).
    ignorar=('maturity',) etc.: ablacao de fontes (avaliar_rotulo.py)."""
    c = _contexto_decisao(r, usar_sinopse, ignorar)
    for _id, _desc, cond, rotulo, conf, fontes in REGRAS_DECISAO:
        if cond(c):
            return rotulo, conf, fontes(c)
    raise RuntimeError("nenhuma regra casou (R8 deveria ser universal)")


def tabela_regras() -> pd.DataFrame:
    """Tabela das regras para o relatorio (ordem = prioridade)."""
    return pd.DataFrame([{"Regra": i, "Condicao": d, "Rotulo": r, "Confianca": c}
                         for i, d, _, r, c, _ in REGRAS_DECISAO])


def rotular(df: pd.DataFrame, pop_pct: float = 50.0, exigir_cw: bool = False,
            ignorar: tuple = ()) -> pd.DataFrame:
    """exigir_cw=True reproduz a definicao antiga de Cobertura_Boa (exigia Content_Warnings preenchido);
    serve para analise de sensibilidade. ignorar=('maturity',...) remove fontes (ablacao)."""
    d = df.copy()
    d["_obra"] = chave_obra(d)
    for nome, fn in (("tags", voto_tags), ("vocab", voto_vocab), ("sinopse", voto_sinopse),
                     ("maturity", voto_maturity), ("cw", voto_cw)):
        d[f"V_{nome}"] = d.apply(fn, axis=1)

    eixos_df = d.apply(eixos_sensiveis_obra, axis=1, result_type="expand")
    for col_eixo in eixos_df.columns:
        d[col_eixo] = eixos_df[col_eixo]

    d["Via_Termo_Conteudo"] = (d["Tipos_Origem"].fillna("").astype(str).str.contains("termo:conteudo")
                               if "Tipos_Origem" in d.columns else False)
    d["YA"] = d.apply(eh_ya, axis=1)
    d["_users"] = pd.to_numeric(d.get("Qtd_Usuarios_Hardcover"), errors="coerce")
    tem = lambda c: d[c].fillna("").astype(str).ne("") if c in d.columns else pd.Series(False, index=d.index)
    limiar = d.drop_duplicates("_obra")["_users"].quantile(pop_pct / 100)
    # Cobertura_Boa: comunidade do Hardcover preencheu a folksonomia geral (Generos, Moods, Tags_Gerais)
    # e ha leitores suficientes (>= limiar). Por padrao NAO exige Content_Warnings: obras leves
    # legitimamente nao tem gatilhos. `exigir_cw=True` reproduz a definicao antiga (sensibilidade).
    d["Cobertura_Boa"] = ((d["_users"] >= limiar) & tem("Generos_Hardcover")
                          & tem("Moods_Hardcover") & tem("Tags_Gerais_Hardcover"))
    if exigir_cw:
        d["Cobertura_Boa"] &= tem("Content_Warnings")

    cols_eixos = ["Flag_Eixo_Sexo", "Flag_Eixo_Violencia", "Flag_Eixo_Drogas_Alcool", "Flag_Eixo_Abuso_Gatilhos"]
    agg = {f"V_{f}": agg_votos for f in FONTES}
    agg.update({
        "YA": "max",
        "Cobertura_Boa": "max",
        "_users": "first",
        "Via_Termo_Conteudo": "max",
        **{c: "max" for c in cols_eixos},
    })
    if "Estratos_Origem" in d.columns:
        agg["Estratos_Origem"] = "first"
    ob = d.groupby("_obra").agg(agg)
    res = ob.apply(lambda r: decidir(r, ignorar=ignorar), axis=1, result_type="expand")
    res.columns = ["Sensivel_Label", "Sensivel_Label_Confianca", "Sensivel_Label_Fontes"]
    res_ns = ob.apply(lambda r: decidir(r, usar_sinopse=False, ignorar=ignorar), axis=1, result_type="expand")
    res["Sensivel_Label_sem_sinopse"] = res_ns[0]
    res["Sensivel_Label_sem_sinopse_Confianca"] = res_ns[1]
    ob = ob.join(res)

    cols_novas = (
        [f"V_{f}" for f in FONTES]
        + cols_eixos
        + ["Cobertura_Boa", "YA", "Via_Termo_Conteudo", "Sensivel_Label", "Sensivel_Label_Confianca",
           "Sensivel_Label_Fontes", "Sensivel_Label_sem_sinopse", "Sensivel_Label_sem_sinopse_Confianca"]
    )
    d = d.drop(columns=[f"V_{f}" for f in FONTES] + cols_eixos + ["Cobertura_Boa", "YA", "Via_Termo_Conteudo"])
    d = d.merge(ob[cols_novas], left_on="_obra", right_index=True, how="left")
    d.attrs["pop_limiar"] = float(limiar)
    return d


def resumo(d: pd.DataFrame, pop_pct: float) -> None:
    C.garantir_dirs()
    ob = d.drop_duplicates("_obra")
    n = len(ob)
    caminho_resumo = os.path.join(C.OUT_REPORTS, "rotulagem_resumo.txt")
    with open(caminho_resumo, "w", encoding="utf-8") as f:
        f.write("=" * 90 + f"\nROTULAGEM TRIANGULADA DE CONTEÚDO INADEQUADO/SENSÍVEL (v3.1) | {datetime.now():%Y-%m-%d %H:%M:%S}\n" + "=" * 90 + "\n")
        f.write(f"Edições: {len(d)} | Obras: {n} | limiar de cobertura: >= {d.attrs.get('pop_limiar', float('nan')):.0f} leitores "
                f"(percentil {pop_pct:.0f}) + Gêneros + Moods + Tags Gerais preenchidos\n\n")
        f.write("RÓTULO FINAL (por obra)\n")
        f.write(ob["Sensivel_Label"].value_counts().to_string() + "\n\n")
        f.write("PREVALÊNCIA POR EIXO DE CONTEÚDO SENSÍVEL/PROBLEMÁTICO (por obra)\n")
        for col_eixo, nome_eixo in (
            ("Flag_Eixo_Sexo", "Sexo / Erotismo / Violência Sexual"),
            ("Flag_Eixo_Violencia", "Violência / Crime / Brutalidade"),
            ("Flag_Eixo_Drogas_Alcool", "Drogas / Bebidas Alcoólicas / Vício"),
            ("Flag_Eixo_Abuso_Gatilhos", "Abuso / Relacionamento Tóxico / Saúde Mental"),
        ):
            if col_eixo in ob.columns:
                k = int(ob[col_eixo].sum())
                f.write(f"  {nome_eixo:<44}: {k}/{n} ({100 * k / max(n, 1):.1f}%)\n")
        f.write("\n")
        f.write("Rótulo x confiança ('circular' = único voto alto era a sinopse de obra achada via termo:conteudo):\n"
                + pd.crosstab(ob["Sensivel_Label"], ob["Sensivel_Label_Confianca"]).to_string() + "\n\n")
        f.write("RÓTULO COM SINOPSE x RÓTULO SEM SINOPSE (use o 2º como alvo ao treinar/avaliar o módulo de texto: "
                "a sinopse é a entrada do módulo e o léxico dela vazaria para o alvo)\n")
        f.write(pd.crosstab(ob["Sensivel_Label"], ob["Sensivel_Label_sem_sinopse"]).to_string() + "\n\n")
        f.write("VOTOS POR FONTE (obra; vazio = abstenção)\n")
        for fn in FONTES:
            f.write(f"  V_{fn:<9}: {ob[f'V_{fn}'].fillna('abstem').value_counts().to_dict()}\n")
        f.write(f"  YA={int(ob['YA'].sum())} | Cobertura_Boa={int(ob['Cobertura_Boa'].sum())} | "
                f"YA & Cobertura_Boa={int((ob['YA'] & ob['Cobertura_Boa']).sum())}\n\n")
        f.write("CIRCULARIDADE: Via_Termo_Conteudo x Sensivel_Label (sinopse dessas obras não conta como independente)\n")
        f.write(pd.crosstab(ob["Via_Termo_Conteudo"], ob["Sensivel_Label"]).to_string() + "\n\n")
        f.write("CONCORDÂNCIA ENTRE FONTES (obras com voto em ambas; linhas=tags, colunas=sinopse)\n")
        ambos = ob.dropna(subset=["V_tags", "V_sinopse"])
        f.write((pd.crosstab(ambos["V_tags"], ambos["V_sinopse"]).to_string() if len(ambos) else "  nenhuma obra com voto em ambas") + "\n\n")
        if "Estratos_Origem" in ob.columns:
            f.write("SANIDADE POR ESTRATO (face validity; NÃO é voto: estratos foram critério de amostragem)\n")
            e = ob.assign(Estrato=ob["Estratos_Origem"].fillna("(fora_das_autoras)").str.split(r" \| ")).explode("Estrato")
            f.write(pd.crosstab(e["Estrato"], e["Sensivel_Label"]).to_string() + "\n\n")
        q = pd.qcut(ob["_users"].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"]) if ob["_users"].notna().sum() >= 8 else None
        if q is not None:
            f.write("RÓTULO x QUARTIL DE POPULARIDADE\n")
            f.write(pd.crosstab(q, ob["Sensivel_Label"]).to_string() + "\n\n")

    conf = ob[ob["Sensivel_Label_Confianca"] == "conflito"]
    cols = [c for c in ("ISBN_PT", "Titulo_PT", "Autor_PT", "V_tags", "V_vocab", "V_sinopse", "V_maturity", "V_cw",
                        "Sensivel_Label_Fontes") if c in conf.columns]
    conf[cols].to_csv(os.path.join(C.OUT_AUDITS, "rotulagem_conflitos.csv"), index=False, encoding="utf-8")


def executar_rotulagem(entrada: Path = Path(C.OUT_LIVROS), saida: Path = Path(C.OUT_LIVROS_ROTULADO),
                       ano_min: int = C.ANO_MIN_PADRAO, pop_pct: float = C.POP_PCT_PADRAO) -> pd.DataFrame:
    df_bruto = pd.read_csv(entrada)
    df, meta = filtrar_escopo(df_bruto, ano_min)
    print(f"Filtro: {meta['n_inicial']} -> {meta['n_final']} edições "
          f"(ano < {ano_min}: {meta['n_antes_ano_min']}; sem ano: {meta['n_sem_ano']}; "
          f"formato excluído: {meta['n_formato_excluido']})")
    d = rotular(df, pop_pct)
    resumo(d, pop_pct)
    saida.parent.mkdir(parents=True, exist_ok=True)
    d.drop(columns=["_obra", "_users"], errors="ignore").to_csv(saida, index=False, encoding="utf-8")
    print(f"Rótulo salvo em {saida}\nResumo: {os.path.join(C.OUT_REPORTS, 'rotulagem_resumo.txt')}")
    print(d.drop_duplicates("_obra")["Sensivel_Label"].value_counts().to_string())
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description="Rotulagem triangulada de conteúdo sensível (v3.1)")
    ap.add_argument("--entrada", type=Path, default=Path(C.OUT_LIVROS))
    ap.add_argument("--saida", type=Path, default=Path(C.OUT_LIVROS_ROTULADO))
    ap.add_argument("--ano-min", type=int, default=C.ANO_MIN_PADRAO, help="ano mínimo da edição PT-BR (igual à EDA)")
    ap.add_argument("--pop-pct", type=float, default=C.POP_PCT_PADRAO, help="percentil de leitores p/ considerar obra bem coberta")
    a = ap.parse_args()
    executar_rotulagem(a.entrada, a.saida, a.ano_min, a.pop_pct)


if __name__ == "__main__":
    main()