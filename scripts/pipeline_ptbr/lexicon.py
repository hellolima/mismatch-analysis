"""Lexico unico dos 4 eixos de conteudo sensivel/problematico (fonte da verdade).

Antes, o vocabulario estava triplicado (labeling.RE_CW_*, rotulagem.RE_ALTO/RE_VOC_ALTO e regexes
inline em rotulagem.eixos_sensiveis_obra). Agora tudo deriva de `EIXOS`, e o lexico pode ser
exportado como tabela (apendice do TCC) via `exportar_lexico_csv()`.

Cada eixo tem tres listas:
  cw    : termos de Content Warning / tags do Hardcover (ingles, texto sem acento, casados por \\b)
  tags  : vocabulario livre de Tags_Gerais/Generos (ingles) -- complementa `cw` no voto de vocabulario
  pt    : termos da sinopse brasileira (portugues sem acento). Podem conter regex simples.

Decisoes de taxonomia (documentar no texto):
  * Violencia SEXUAL (rape, sexual assault, estupro) conta em DOIS eixos: 'sexo' (conteudo sexual) e
    'abuso' (gatilho). Justificativa: para o leitor, e simultaneamente conteudo sexual e trauma.
  * 'war' e 'death' NAO entram em violencia (disparam em qualquer fantasia epica / drama); apenas
    'graphic violence', 'gun violence', 'murder', 'torture', 'gore', 'blood' etc.
  * 'dark romance' e tratado como marcador de subgenero com conteudo sexual E temas sombrios: entra em 'sexo'.
"""
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

EIXOS: Dict[str, Dict[str, object]] = {
    "sexo": {
        "rotulo": "Sexo / Erotismo / Spice",
        "cw": [
            "sexual content", "explicit sexual content", "rape", "sexual assault", "sexual violence",
            "sexual harassment", r"non[\s-]?con", r"dub[\s-]?con",
        ],
        "tags": [
            "spicy", "steamy", r"smut+y?", r"erotic(?:a|ism)?", "erotic stories", r"open[\s-]door",
            "high heat", "explicit", "dark romance",
        ],
        "pt": [
            "conteudo adulto", "conteudo explicito", "conteudo sexual",
            r"cenas? (?:de sexo|explicitas?|quentes|picantes?)", "sexo explicito", r"erotic[oa]s?",
            r"\+ ?18", r"18 ?\+", "maiores de 18", "dark romance", "smut", "spicy", "steamy",
            r"torrid[oa]", "alta voltagem", "hot romance", "estupro", "assedio sexual", "violencia sexual",
        ],
    },
    "violencia": {
        "rotulo": "Violência / Crime / Brutalidade",
        "cw": [
            "violence", "graphic violence", "gun violence", "murder", "homicide", "blood", "gore",
            "torture", "kidnapping", "abduction", "confinement", "animal cruelty", "massacre",
        ],
        "tags": ["bloody", "serial killer", "mafia"],
        "pt": [
            r"violencia (?:extrema|fisica|grafica|psicologica|domestica)", "agressao", "tortura",
            "sequestro", "carcere privado", r"\bgore\b", "carnificina", "mutilacao", "assassinato brutal",
            "canibalismo", "massacre", "chacina",
        ],
    },
    "drogas_alcool": {
        "rotulo": "Drogas / Bebidas Alcoólicas / Vício",
        "cw": [
            "alcohol", "alcoholism", "drinking", r"drunk\w*", "drug use", "drug abuse", "drugs",
            "addiction", "substance abuse", "overdose", "narcotics",
        ],
        "tags": [],
        "pt": [
            "uso de drogas", "trafico de drogas", "abuso de drogas", "drogas ilicitas",
            r"substancias? ilicitas?", "dependencia quimica", "overdose", "entorpecentes", "alcoolismo",
            "abuso de alcool", r"bebidas? alcoolicas?", "embriaguez", r"vicio em (?:drogas|alcool|bebida)",
        ],
    },
    "abuso_gatilhos": {
        "rotulo": "Abuso / Relacionamento Tóxico / Saúde Mental",
        "cw": [
            "child abuse", "domestic abuse", "emotional abuse", "physical abuse", "abuse",
            r"self[\s-]?harm", "suicidal thoughts", "suicide", "stalking", "toxic relationship",
            "gaslighting", "bullying", "rape", "sexual assault", "sexual violence",
        ],
        "tags": ["trigger warning"],
        "pt": [
            r"abuso (?:fisico|emocional|psicologico|sexual|infantil|domestico)", "relacionamento toxico",
            "suicidio", "ideacao suicida", "automutilacao", "estupro", "assedio sexual", "violencia sexual",
        ],
    },
}

# Polo NEGATIVO (conteudo leve / fechado). Unico para todos os eixos.
BAIXO_EN: List[str] = [
    r"closed[\s-]door", "clean romance", "sweet romance", "fade to black", "low heat", "no spice",
    "no sex", r"family[\s-]friendly", "all ages",
]
BAIXO_PT: List[str] = [
    "romance clean", "clean romance", "sweet romance", r"sem cenas? (?:de sexo|explicitas?)",
    r"sem conteudo (?:adulto|explicito|sexual)", "porta fechada", "closed door", "fade to black",
    "para todas as idades",
]

ORDEM_EIXOS = ("sexo", "violencia", "drogas_alcool", "abuso_gatilhos")
COLUNA_EIXO = {
    "sexo": "Flag_Eixo_Sexo",
    "violencia": "Flag_Eixo_Violencia",
    "drogas_alcool": "Flag_Eixo_Drogas_Alcool",
    "abuso_gatilhos": "Flag_Eixo_Abuso_Gatilhos",
}


def _compilar(termos: List[str], borda: bool = True) -> "re.Pattern":
    corpo = "|".join(termos)
    return re.compile(rf"\b(?:{corpo})\b" if borda else corpo)


def regex_cw(eixo: str) -> "re.Pattern":
    return _compilar(EIXOS[eixo]["cw"])


def regex_vocab(eixo: str) -> "re.Pattern":
    return _compilar(EIXOS[eixo]["cw"] + EIXOS[eixo]["tags"])


def regex_pt(eixo: str) -> "re.Pattern":
    return _compilar(EIXOS[eixo]["pt"], borda=False)


# Regexes prontas (uniao de todos os eixos)
RE_CW_SENSIVEL = _compilar([t for e in ORDEM_EIXOS for t in EIXOS[e]["cw"]])
RE_VOC_ALTO = _compilar([t for e in ORDEM_EIXOS for t in EIXOS[e]["cw"] + EIXOS[e]["tags"]])
RE_VOC_BAIXO = _compilar(BAIXO_EN)
RE_ALTO = _compilar([t for e in ORDEM_EIXOS for t in EIXOS[e]["pt"]], borda=False)
RE_BAIXO = _compilar(BAIXO_PT, borda=False)
RE_POR_EIXO = {e: {"cw": regex_cw(e), "vocab": regex_vocab(e), "pt": regex_pt(e)} for e in ORDEM_EIXOS}


def tabela_lexico() -> pd.DataFrame:
    """Lexico em formato longo: eixo, fonte (cw/tags/pt/baixo), termo."""
    rows = []
    for e in ORDEM_EIXOS:
        for fonte in ("cw", "tags", "pt"):
            for t in EIXOS[e][fonte]:
                rows.append({"Eixo": e, "Rotulo_Eixo": EIXOS[e]["rotulo"], "Fonte": fonte, "Termo": t})
    for t in BAIXO_EN:
        rows.append({"Eixo": "baixo", "Rotulo_Eixo": "Polo negativo (leve/fechado)", "Fonte": "tags", "Termo": t})
    for t in BAIXO_PT:
        rows.append({"Eixo": "baixo", "Rotulo_Eixo": "Polo negativo (leve/fechado)", "Fonte": "pt", "Termo": t})
    return pd.DataFrame(rows)


def exportar_lexico_csv(caminho: Path) -> pd.DataFrame:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    t = tabela_lexico()
    t.to_csv(caminho, index=False, encoding="utf-8")
    return t
