"""
rotulagem.py  (colocar em scripts/pipeline_ptbr/)   [v3.1]

Rótulo de CONTEÚDO INADEQUADO / SENSÍVEL por triangulação de fontes, com abstenção explícita.

Princípio: cada fonte de evidência vota 'alto', 'baixo' ou se ABSTÉM (None). Ausência de evidência
NUNCA vira 'baixo' por padrão (a ausência de tag é dado faltante não aleatório: depende de popularidade).

MUDANÇAS DA v3.1 EM RELAÇÃO À v3:
  1. Nomenclatura: o rótulo agregado deixa de se chamar Spice_Label* e passa a Sensivel_Label*
     (Sensivel_Label, Sensivel_Label_Confianca, Sensivel_Label_Fontes, Sensivel_Label_sem_sinopse,
     Sensivel_Label_sem_sinopse_Confianca). O rótulo cobre sexual + violência + temas sensíveis, então 'spice'
     seria conceitualmente incorreto. Os SINAIS DE ORIGEM do Hardcover (Spice_Nivel, Spice_Status,
     Flag_Spice_Tag, Spice_Tags) NÃO foram renomeados: eles medem especificamente conteúdo sexual.
  2. Léxico PT (RE_ALTO): + gore, carnificina, mutilacao, assassinato brutal, canibalismo.
     REMOVIDO 'trauma' (amplo demais: "superou um trauma" em romance leve virava voto alto).
     Não incluído de propósito: 'terror psicologico' (é gênero, não intensidade).
  3. Léxico EN (RE_VOC_ALTO): + gore, bloody, graphic violence, massacre, serial killer.
     Não incluídos de propósito: horror, murder, psychological thriller (são TAGS DE GÊNERO; incluí-los
     reintroduziria rótulo por gênero, contrariando o desenho do projeto).
  4. Léxico de adequação: RE_VOC_BAIXO + family friendly, all ages; RE_BAIXO (PT) + para todas as idades.
     Não incluídos de propósito: cozy mystery, misterio leve, livro familia (gênero/ambíguos; ruído em
     'baixo' contamina os negativos do POC II).
  5. eh_ya: match por TAG delimitada (antes: substring em texto livre, que casava "Children of Blood and
     Bone", "juvenile delinquency" etc.). 'new adult' não casa com 'young adult'.
  voto_tags permanece inalterada: para não-romance Spice_Nivel é nulo e a fonte se abstém.

  Herdado da v3: provavel_baixo exige Content_Warnings preenchido; circularidade via termo:conteudo;
  Spice_Nivel 1 abstém; rótulo sem o voto da sinopse (alvo seguro para o módulo de texto do POC II).

FONTES (por edição; agregadas por OBRA; votos divergentes entre edições = 'conflito'):
  V_tags      Hardcover: Spice_Nivel >= 2 -> alto; 0 ou Closed_Door -> baixo; 1 -> abstém
  V_vocab     vocabulário lido em Tags_Gerais/Generos: escala "N peppers"/"spice level N"
              (N >= PEPPER_ALTO -> alto; N <= PEPPER_BAIXO -> baixo) + termos de conteúdo explícito,
              violência e temas sensíveis -> alto; closed door, clean/sweet romance, low heat,
              family friendly, all ages -> baixo
  V_sinopse   léxico PT na Sinopse_PT: termos explícitos, violência e abuso -> alto; 'clean/porta fechada',
              'para todas as idades' -> baixo
  V_maturity  Maturity_GB == MATURE -> alto (senão abstém; campo mantido por retrocompatibilidade)
  V_cw        Flag_CW_Sexual_Content / CWs Sensíveis -> alto (FORTE, pois indica conteúdo sensível/inadequado)

FAMÍLIAS INDEPENDENTES (para 'forte'): comunidade (= tags + vocab, mesma origem: leitores do Hardcover),
sinopse, maturity. CW conta como validador direto para rótulo alto.

CIRCULARIDADE: obras que entraram no seed via 'termo:conteudo' (Via_Termo_Conteudo=True) casam com o léxico
da sinopse POR CONSTRUÇÃO (a busca no Google Books usou esse termo). O voto de sinopse delas continua
registrado, mas NÃO conta como família independente e, sozinho, NÃO sustenta rótulo (-> circular).

DECISÃO POR OBRA:
  famílias divergentes (alto x baixo) ou voto 'conflito'          -> indeterminado / conflito (revisão manual)
  único voto alto = sinopse de obra via termo:conteudo            -> indeterminado / circular
  >= 2 famílias independentes votam alto OU CW ativo              -> alto / forte
  1 família alta                                                  -> alto / fraca
  >= 2 famílias votam baixo                                       -> baixo / forte
  1 família baixa                                                 -> baixo / fraca
  sem voto, mas obra bem coberta (inclui CW) + YA                 -> provavel_baixo / nao_validado
  caso contrário                                                  -> indeterminado / sem_evidencia

'provavel_baixo' é uma INFERÊNCIA a partir do gênero YA (nenhuma fonte votou baixo): não use como negativo
sem validação humana. 'baixo' exige evidência própria.

LIMITAÇÃO CONHECIDA: Content_Warnings cobre só ~25% das obras (0-2% no quartil menos popular). Para não-romance
de nicho, V_tags e V_vocab se abstêm e o rótulo tende a 'indeterminado/sem_evidencia'. V_cw hoje só detecta
conteúdo sexual; CWs de violência/abuso/self harm ainda NÃO geram voto.

NÃO são votos: Estratos_Origem e Tipos_Origem (critérios de AMOSTRAGEM; só sanidade/diagnóstico).

Saídas:
  data/processed/livros_ptbr_v3_rotulado.csv   (todas as colunas + colunas do rótulo; já FILTRADO)
  outputs/rotulagem_resumo.txt, outputs/rotulagem_conflitos.csv

Uso:  python scripts/pipeline_ptbr/rotulagem.py [--entrada CSV] [--ano-min 2015] [--pop-pct 50]
"""

import re
import argparse
import unicodedata
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
ENTRADA = RAIZ / "data" / "processed" / "livros_ptbr_v3.csv"
SAIDA = RAIZ / "data" / "processed" / "livros_ptbr_v3_rotulado.csv"
OUT = RAIZ / "outputs"

FONTES = ('tags', 'vocab', 'sinopse', 'maturity', 'cw')
PEPPER_ALTO = 3    # escala 0-5: >=3 'alto'
PEPPER_BAIXO = 1   # <=1 'baixo'; 2 = abstém (ambíguo)


def _norm(s: object) -> str:
    """minúsculas, sem acento."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    t = unicodedata.normalize("NFD", str(s).lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def _b(x: object) -> bool:
    return str(x).strip().lower() in ("true", "1", "1.0")


def _v(x: object) -> Optional[str]:
    """NaN/None -> None (abstenção); string -> string."""
    return x if isinstance(x, str) else None


# Léxicos da sinopse (texto sem acento): Conteúdo Sexual + Violência + Temas Sensíveis/Inadequados
RE_ALTO = re.compile(
    r"conteudo adulto|conteudo explicito|conteudo sexual|cenas? (?:de sexo|explicitas?|quentes|picantes?)|"
    r"sexo explicito|erotic[oa]s?|\+ ?18|18 ?\+|maiores de 18|dark romance|smut|spicy|steamy|torrid[oa]|"
    r"alta voltagem|hot romance|"
    r"violencia (?:extrema|fisica|grafica|psicologica)|abuso (?:fisico|emocional|psicologico|sexual)|"
    r"relacionamento toxico|agressao|tortura|sequestro|suicidio|automutilacao|uso de drogas|"
    r"\bgore\b|carnificina|mutilacao|assassinato brutal|canibalismo"
)
RE_BAIXO = re.compile(
    r"romance clean|clean romance|sweet romance|sem cenas? (?:de sexo|explicitas?)|"
    r"sem conteudo (?:adulto|explicito|sexual)|porta fechada|closed door|fade to black|"
    r"para todas as idades"
)

# Vocabulário de tags do Hardcover (inglês, texto sem acento): Spice + Content Warnings
RE_PEPPER = re.compile(r"(?<!\d)(\d)\s*peppers?\b|\bspice[\s\-]*(?:level)?[\s\-]*(\d)\b")
RE_VOC_ALTO = re.compile(
    r"\b("
    r"spicy|steamy|smut+y?|erotic(?:a|ism)?|erotic stories|open[\s\-]door|high heat|explicit|"
    r"dark romance|trigger warning|violence|abuse|non\-con|dub\-con|self\-harm|suicide|addiction|"
    r"gore|bloody|graphic violence|massacre|serial killer"
    r")\b"
)
RE_VOC_BAIXO = re.compile(
    r"\b(closed[\s\-]door|clean romance|sweet romance|fade to black|low heat|no spice|no sex|"
    r"family[\s\-]friendly|all ages)\b"
)

# YA / infantil: casa a TAG inteira (não substring de texto livre)
RE_YA_TAG = re.compile(
    r"^(?:young adult\b.*|middle grade\b.*|juvenile\b.*|children'?s?(?: books?| fiction| literature)?|"
    r"infantil\b.*|infantojuvenil\b.*)$"
)


def voto_tags(r: pd.Series) -> Optional[str]:
    nivel = pd.to_numeric(r.get('Spice_Nivel'), errors='coerce')
    if pd.notna(nivel):
        if nivel >= 2:
            return 'alto'
        if nivel == 0:
            return 'baixo'
        # nivel == 1 (sensual/implícito, ex.: 'sexy'): não é evidência de spice BAIXO -> abstém e cai nas flags
    if _b(r.get('Flag_Spice_Tag')):
        return 'alto'
    if _b(r.get('Flag_Closed_Door_Tag')):
        return 'baixo'
    return None


def voto_vocab(r: pd.Series) -> Optional[str]:
    tags = []
    for col in ('Tags_Gerais_Hardcover', 'Generos_Hardcover'):
        s = r.get(col)
        if isinstance(s, str):
            tags += [_norm(t).strip() for t in s.split('|') if t.strip()]
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
        return 'conflito'
    return 'alto' if altos else ('baixo' if baixos else None)


def voto_sinopse(r: pd.Series) -> Optional[str]:
    t = _norm(r.get('Sinopse_PT'))
    if not t:
        return None
    a, b = bool(RE_ALTO.search(t)), bool(RE_BAIXO.search(t))
    if a and b:
        return 'conflito'
    return 'alto' if a else ('baixo' if b else None)


def voto_maturity(r: pd.Series) -> Optional[str]:
    # Como Maturity_GB não foi extraído no primeiro script, retorna None por padrão se ausente
    val = r.get('Maturity_GB')
    if pd.isna(val) or val is None:
        return None
    return 'alto' if str(val).strip().upper() == 'MATURE' else None


def voto_cw(r: pd.Series) -> Optional[str]:
    # Avalia flags de gatilho/sensíveis existentes
    has_sexual = _b(r.get('Flag_CW_Sexual_Content'))
    has_cw_gen = _b(r.get('Flag_Has_CW')) if 'Flag_Has_CW' in r else False
    return 'alto' if (has_sexual or has_cw_gen) else None


def agg_votos(s: pd.Series) -> Optional[str]:
    """Agrega votos das edições de uma obra: divergência = conflito."""
    v = {x for x in s if isinstance(x, str)}
    if not v:
        return None
    if 'conflito' in v or {'alto', 'baixo'} <= v:
        return 'conflito'
    return next(iter(v))


def eh_ya(r: pd.Series) -> bool:
    """YA/infantil por TAG delimitada em Generos_Hardcover ('|') e Categorias_GB ('|', ',', ';').
    Não usa substring em texto livre: 'children of blood and bone' ou 'juvenile delinquency' não casam."""
    tags = []
    for col, sep in (('Generos_Hardcover', r"\|"), ('Categorias_GB', r"[|,;]")):
        s = _norm(r.get(col))
        if s:
            tags += [t.strip() for t in re.split(sep, s) if t.strip()]
    return any(RE_YA_TAG.match(t) for t in tags)


def _comb(a: Optional[str], b: Optional[str]) -> Optional[str]:
    s = {x for x in (a, b) if x}
    if not s:
        return None
    if 'conflito' in s or s == {'alto', 'baixo'}:
        return 'conflito'
    return next(iter(s))


def decidir(r: pd.Series, usar_sinopse: bool = True):
    """usar_sinopse=False: mesmo rótulo sem o voto da sinopse (para o módulo de texto do POC II)."""
    v = {f: _v(r[f'V_{f}']) for f in FONTES}
    if not usar_sinopse:
        v['sinopse'] = None
    via = bool(r['Via_Termo_Conteudo'])

    # Famílias independentes
    fam = {
        'comunidade': _comb(v['tags'], v['vocab']),
        'sinopse': v['sinopse'],
        'maturity': v['maturity']
    }

    alto_f = [k for k, x in fam.items() if x == 'alto']
    baixo_f = [k for k, x in fam.items() if x == 'baixo']
    conf_f = [k for k, x in fam.items() if x == 'conflito']

    cw = v['cw'] == 'alto'

    com_voto = '|'.join(f for f in FONTES if v[f] in ('alto', 'baixo', 'conflito'))
    det_alto = '|'.join(f for f in FONTES if v[f] == 'alto')
    det_baixo = '|'.join(f for f in FONTES if v[f] == 'baixo')

    # Se houver conflito entre fontes
    if conf_f or (baixo_f and (alto_f or cw)):
        return 'indeterminado', 'conflito', com_voto

    # Desconsidera sinopse se o livro veio de busca por termo de conteúdo (circularidade)
    indep = [f for f in alto_f if not (f == 'sinopse' and via)]

    # o único voto alto é a sinopse de uma obra achada POR esse termo -> artefato da busca, não evidência
    if alto_f and not indep and not cw:
        return 'indeterminado', 'circular', 'sinopse(via_termo_conteudo)'

    # Se houver 2+ famílias OU flag de CW ativa, considera ALTO FORTE
    if len(indep) >= 2 or cw:
        return 'alto', 'forte', det_alto

    if alto_f:
        return 'alto', 'fraca', det_alto

    if len(baixo_f) >= 2:
        return 'baixo', 'forte', det_baixo

    if baixo_f:
        return 'baixo', 'fraca', det_baixo

    if bool(r['Cobertura_Boa']) and bool(r['YA']):
        return 'provavel_baixo', 'nao_validado', 'cobertura_ya'

    return 'indeterminado', 'sem_evidencia', ''


def filtrar(df: pd.DataFrame, ano_min: int) -> pd.DataFrame:
    """Mesmo filtro da EDA: ano da edição >= ano_min (ano ausente sai) e formatos fora de escopo."""
    n0 = len(df)
    if 'Ano_Publicacao_PT' not in df.columns:
        print("AVISO: Ano_Publicacao_PT ausente; filtro de ano NÃO aplicado.")
        return df
    ano = df['Ano_Publicacao_PT'].astype(str).str.extract(r"(\d{4})")[0].astype(float)
    sem_ano, antes = int(ano.isna().sum()), int((ano < ano_min).sum())
    df = df[ano >= ano_min].copy()
    n_fmt = 0
    if 'Flag_Formato_Excluir' in df.columns:
        ex = df['Flag_Formato_Excluir'].map(_b)
        n_fmt = int(ex.sum())
        df = df[~ex].copy()
    print(f"Filtro: {n0} -> {len(df)} edições (ano < {ano_min}: {antes}; sem ano: {sem_ano}; formato excluído: {n_fmt})")
    return df.reset_index(drop=True)


def rotular(df: pd.DataFrame, pop_pct: float = 50.0) -> pd.DataFrame:
    d = df.copy()
    d['_obra'] = np.where(d['Hardcover_Book_ID'].notna() if 'Hardcover_Book_ID' in d.columns else False,
                          d.get('Hardcover_Book_ID', pd.Series(index=d.index, dtype=object)).astype(str),
                          'isbn:' + d['ISBN_PT'].astype(str))
    for nome, fn in (('tags', voto_tags), ('vocab', voto_vocab), ('sinopse', voto_sinopse),
                     ('maturity', voto_maturity), ('cw', voto_cw)):
        d[f'V_{nome}'] = d.apply(fn, axis=1)
    d['Via_Termo_Conteudo'] = (d['Tipos_Origem'].fillna('').astype(str).str.contains('termo:conteudo')
                               if 'Tipos_Origem' in d.columns else False)
    d['YA'] = d.apply(eh_ya, axis=1)
    d['_users'] = pd.to_numeric(d.get('Qtd_Usuarios_Hardcover'), errors='coerce')
    tem = lambda c: d[c].fillna('').astype(str).ne('') if c in d.columns else pd.Series(False, index=d.index)
    if 'Content_Warnings' not in d.columns:
        print("AVISO: coluna Content_Warnings ausente; Cobertura_Boa será sempre False e NÃO haverá 'provavel_baixo'.")
    limiar = d.drop_duplicates('_obra')['_users'].quantile(pop_pct / 100)
    # a obra precisa ter CW registrados, senão a ausência de CW sexual não diz nada
    d['Cobertura_Boa'] = ((d['_users'] >= limiar) & tem('Moods_Hardcover') & tem('Tags_Gerais_Hardcover')
                          & tem('Content_Warnings'))

    agg = {f'V_{f}': agg_votos for f in FONTES}
    agg.update({'YA': 'max', 'Cobertura_Boa': 'max', '_users': 'first', 'Via_Termo_Conteudo': 'max'})
    if 'Estratos_Origem' in d.columns:
        agg['Estratos_Origem'] = 'first'
    ob = d.groupby('_obra').agg(agg)
    res = ob.apply(decidir, axis=1, result_type='expand')
    res.columns = ['Sensivel_Label', 'Sensivel_Label_Confianca', 'Sensivel_Label_Fontes']
    # rótulo sem o voto da sinopse (alvo seguro para o módulo de texto)
    res_ns = ob.apply(lambda r: decidir(r, usar_sinopse=False), axis=1, result_type='expand')
    res['Sensivel_Label_sem_sinopse'] = res_ns[0]
    res['Sensivel_Label_sem_sinopse_Confianca'] = res_ns[1]
    ob = ob.join(res)

    cols_novas = ([f'V_{f}' for f in FONTES]
                  + ['Cobertura_Boa', 'YA', 'Via_Termo_Conteudo', 'Sensivel_Label', 'Sensivel_Label_Confianca',
                     'Sensivel_Label_Fontes', 'Sensivel_Label_sem_sinopse', 'Sensivel_Label_sem_sinopse_Confianca'])
    d = d.drop(columns=[f'V_{f}' for f in FONTES] + ['Cobertura_Boa', 'YA', 'Via_Termo_Conteudo'])
    d = d.merge(ob[cols_novas], left_on='_obra', right_index=True, how='left')
    d.attrs['pop_limiar'] = float(limiar)
    return d


def resumo(d: pd.DataFrame, pop_pct: float) -> None:
    OUT.mkdir(exist_ok=True)
    ob = d.drop_duplicates('_obra')
    n = len(ob)
    with open(OUT / 'rotulagem_resumo.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 90 + f"\nROTULAGEM TRIANGULADA DE CONTEÚDO INADEQUADO/SENSÍVEL (v3.1) | {datetime.now():%Y-%m-%d %H:%M:%S}\n" + "=" * 90 + "\n")
        f.write(f"Edições: {len(d)} | Obras: {n} | limiar de cobertura: >= {d.attrs.get('pop_limiar', float('nan')):.0f} leitores "
                f"(percentil {pop_pct:.0f}) + Moods + Tags Gerais + Content Warnings preenchidos\n\n")
        f.write("RÓTULO FINAL (por obra)\n")
        f.write(ob['Sensivel_Label'].value_counts().to_string() + "\n\n")
        f.write("Rótulo x confiança ('circular' = único voto alto era a sinopse de obra achada via termo:conteudo):\n"
                + pd.crosstab(ob['Sensivel_Label'], ob['Sensivel_Label_Confianca']).to_string() + "\n\n")
        f.write("RÓTULO COM SINOPSE x RÓTULO SEM SINOPSE (use o 2º como alvo ao treinar/avaliar o módulo de texto: "
                "a sinopse é a entrada do módulo e o léxico dela vazaria para o alvo)\n")
        f.write(pd.crosstab(ob['Sensivel_Label'], ob['Sensivel_Label_sem_sinopse']).to_string() + "\n\n")
        f.write("VOTOS POR FONTE (obra; vazio = abstenção)\n")
        for fn in FONTES:
            f.write(f"  V_{fn:<9}: {ob[f'V_{fn}'].fillna('abstem').value_counts().to_dict()}\n")
        f.write(f"  YA={int(ob['YA'].sum())} | Cobertura_Boa={int(ob['Cobertura_Boa'].sum())} | "
                f"YA & Cobertura_Boa={int((ob['YA'] & ob['Cobertura_Boa']).sum())}\n\n")
        f.write("CIRCULARIDADE: Via_Termo_Conteudo x Sensivel_Label (sinopse dessas obras não conta como independente)\n")
        f.write(pd.crosstab(ob['Via_Termo_Conteudo'], ob['Sensivel_Label']).to_string() + "\n\n")
        f.write("CONCORDÂNCIA ENTRE FONTES (obras com voto em ambas; linhas=tags, colunas=sinopse)\n")
        ambos = ob.dropna(subset=['V_tags', 'V_sinopse'])
        f.write((pd.crosstab(ambos['V_tags'], ambos['V_sinopse']).to_string() if len(ambos) else "  nenhuma obra com voto em ambas") + "\n\n")
        if 'Estratos_Origem' in ob.columns:
            f.write("SANIDADE POR ESTRATO (face validity; NÃO é voto: estratos foram critério de amostragem)\n")
            e = ob.assign(Estrato=ob['Estratos_Origem'].fillna('(fora_das_autoras)').str.split(r' \| ')).explode('Estrato')
            f.write(pd.crosstab(e['Estrato'], e['Sensivel_Label']).to_string() + "\n\n")
        q = pd.qcut(ob['_users'].rank(method='first'), 4, labels=['Q1', 'Q2', 'Q3', 'Q4']) if ob['_users'].notna().sum() >= 8 else None
        if q is not None:
            f.write("RÓTULO x QUARTIL DE POPULARIDADE\n")
            f.write(pd.crosstab(q, ob['Sensivel_Label']).to_string() + "\n\n")

    conf = ob[ob['Sensivel_Label_Confianca'] == 'conflito']
    cols = [c for c in ('ISBN_PT', 'Titulo_PT', 'Autor_PT', 'V_tags', 'V_vocab', 'V_sinopse', 'V_maturity', 'V_cw',
                        'Sensivel_Label_Fontes') if c in conf.columns]
    conf[cols].to_csv(OUT / 'rotulagem_conflitos.csv', index=False, encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--entrada', type=Path, default=ENTRADA)
    ap.add_argument('--saida', type=Path, default=SAIDA)
    ap.add_argument('--ano-min', type=int, default=2015, help='ano mínimo da edição PT-BR (igual à EDA)')
    ap.add_argument('--pop-pct', type=float, default=50.0, help='percentil de leitores p/ considerar obra bem coberta')
    a = ap.parse_args()
    df = filtrar(pd.read_csv(a.entrada), a.ano_min)
    d = rotular(df, a.pop_pct)
    resumo(d, a.pop_pct)
    d.drop(columns=['_obra', '_users']).to_csv(a.saida, index=False, encoding='utf-8')
    print(f"Rótulo salvo em {a.saida}\nResumo: {OUT / 'rotulagem_resumo.txt'}")
    print(d.drop_duplicates('_obra')['Sensivel_Label'].value_counts().to_string())


if __name__ == "__main__":
    main()