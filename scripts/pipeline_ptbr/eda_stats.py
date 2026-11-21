"""Camada de logica estatistica e processamento de texto da EDA (sem dependencia de matplotlib).

Separa os calculos (estatistica descritiva, IC de Wilson, Bootstrap, Spearman, n-gramas,
canonizacao de editoras e engenharia de features) da geracao de figuras e relatorios.
"""
import re
import logging
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .common import limpar_titulo, sem_acento
from .publishers import chave_editora
from .dataset import filtrar_escopo, deduplicar_por_obra
from .labeling import tags_por_categoria, spice_da_obra

log = logging.getLogger("pipeline.eda.stats")

SEED = 42

# Tags do Hardcover que parecem respostas a perguntas estruturadas de review (pares polares), nao
# folksonomia livre. Separadas de Tags_Gerais para nao esconder os tropos reais.
TAGS_ESTRUTURADAS = {
    "strong character development", "weak character development",
    "loveable characters", "unloveable characters",
    "diverse characters", "not diverse characters",
    "a mix driven", "character driven", "plot driven",
}

# =====================================================================
# STOPWORDS (PT, EN e Ruido Editorial)
# =====================================================================
STOPWORDS_PT = {
    "o", "a", "os", "as", "um", "uma", "uns", "umas",
    "de", "em", "para", "com", "por", "sobre", "até", "desde", "entre", "durante",
    "sem", "perante", "mediante", "segundo", "conforme", "consoante",
    "e", "ou", "mas", "porém", "contudo", "todavia", "entretanto", "portanto", "logo",
    "pois", "que", "se", "como",
    "ser", "estar", "ter", "haver", "dever", "poder", "ir", "vir",
    "sou", "és", "é", "somos", "sois", "são",
    "estou", "estás", "está", "estamos", "estais", "estão",
    "tenho", "tens", "tem", "temos", "tendes", "têm",
    "eu", "tu", "ele", "ela", "nós", "vós", "eles", "elas",
    "me", "te", "nos", "vos", "meu", "teu", "seu", "nosso", "vosso",
    "minha", "tua", "sua", "nossa", "vossa", "meus", "teus", "seus", "nossos", "vossos",
    "minhas", "tuas", "suas", "nossas", "vossas",
    "este", "esse", "aquele", "esta", "essa", "aquela",
    "isto", "isso", "aquilo", "estes", "esses", "aqueles",
    "da", "do", "das", "dos", "vai", "vão", "será", "serão",
    "foi", "fomos", "foram", "era", "éramos", "eram",
    "muito", "pouco", "todo", "nenhum", "algum", "outro", "mesmo", "próprio",
    "mais", "menos", "bem", "mal", "cá", "aí", "lá",
    "já", "ainda", "sempre", "nunca", "jamais", "talvez", "também", "nem", "sequer",
    "aqui", "agora", "hoje", "ontem", "amanhã", "quando", "onde", "porque", "quem",
    "no", "na", "nas", "ao", "aos", "à", "às", "pelo", "pela", "pelos", "pelas",
    "num", "numa", "nuns", "numas", "dum", "duma", "duns", "dumas",
    "dele", "dela", "deles", "delas", "neste", "nesta", "nestes", "nestas",
    "nesse", "nessa", "nesses", "nessas", "naquele", "naquela", "naqueles", "naquelas",
    "nisto", "nisso", "naquilo", "deste", "desta", "destes", "destas",
    "desse", "dessa", "desses", "dessas", "daquele", "daquela", "daqueles", "daquelas",
    "disto", "disso", "daquilo", "àquele", "àquela", "àqueles", "àquelas",
    "lhe", "lhes", "mim", "ti", "si", "consigo", "comigo", "contigo", "conosco", "convosco",
    "você", "vocês", "outra", "outros", "outras", "tal", "tais", "qual", "quais", "quê",
    "cujo", "cuja", "cujos", "cujas", "quanto", "quanta", "quantos", "quantas",
    "alguém", "ninguém", "tudo", "nada", "algo", "cada", "vários", "várias", "alguns", "algumas",
    "nenhuma", "nenhuns", "nenhumas", "toda", "todos", "todas", "certo", "certa",
    "não", "sim", "assim", "apenas", "só", "somente", "depois", "antes", "após", "então",
    "enquanto", "embora", "apesar", "entanto", "tanto", "tão", "quase", "tampouco",
    "ali", "além", "através", "sob", "contra", "ante",
    "inclusive", "sendo", "caso", "dado",
    "pra", "pro", "pras", "pros", "né", "tá",
    "dois", "duas", "três",
    "sido", "fui", "fora", "seja", "sejam", "seria", "serem",
    "estava", "estavam", "esteja", "estive", "estiver",
    "tinha", "tinham", "tenha", "tenham", "tido", "tendo", "teve", "tiver",
    "há", "havia", "houve", "haja",
    "pode", "podem", "podia", "podiam", "possa", "possam", "pôde",
    "deve", "devem", "devia", "vem", "vêm", "veio", "vindo", "vamos", "ia", "iam",
    "quer", "querem", "queria", "queriam", "quis",
}
STOPWORDS_PT = STOPWORDS_PT | {sem_acento(w) for w in STOPWORDS_PT}

STOPWORDS_EN = {
    "the", "of", "and", "to", "in", "on", "at", "for", "with", "a", "an",
    "is", "it", "my", "your", "you", "we", "our", "his", "her", "its",
}

STOPWORDS_EDITORIAL = {
    "livro", "livros", "autora", "autor", "romance", "romances", "best-seller", "bestseller",
    "sucesso", "leitores", "leitor", "leitura", "edição", "edicao", "série", "serie", "volume",
    "história", "historia", "ler", "new", "york", "times", "tiktok", "booktok",
    "publishers", "weekly", "kirkus", "reviews", "goodreads", "amazon",
}
STOPWORDS_EDITORIAL |= {sem_acento(w) for w in STOPWORDS_EDITORIAL}

STOP_TODAS = STOPWORDS_PT | STOPWORDS_EN
RE_TOKEN = re.compile(r"[a-zà-ÿ0-9]+(?:-[a-zà-ÿ0-9]+)*")


# =====================================================================
# FUNCOES ESTATISTICAS E DE NORMALIZACAO
# =====================================================================
def wilson_ic(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Proporcao e IC de Wilson (95%). Adequado para n pequeno, ao contrario do IC de Wald."""
    if n <= 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    centro = (p + z ** 2 / (2 * n)) / den
    marg = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return p, max(0.0, centro - marg), min(1.0, centro + marg)


# ---------------------------------------------------------------------------
# Faixas de popularidade e de nota (cortes FIXOS, pre-especificados)
# ---------------------------------------------------------------------------
# Popularidade = nº de leitores no Hardcover. Cortes por ordem de grandeza (escala log), nao por
# quantil: 76 obras tem <=2 leitores, entao qualquer quartil exige desempate arbitrario entre obras
# indistinguiveis. Tamanhos desiguais sao esperados e o IC de Wilson lida com isso.
POP_CORTES = (0, 10, 100, 1000, np.inf)
POP_ROTULOS = ("<10", "10–99", "100–999", "≥1000")

# Nota media no Hardcover. Exige minimo de avaliacoes: notas com 1–2 votos sao ruido (todas as
# obras com nota 5,0 tem 1 voto). Cortes fixos em torno da massa da distribuicao (3,5–4,25).
NOTA_MIN_AVALIACOES = 10
NOTA_CORTES = (0, 3.5, 4.0, 5.01)
NOTA_ROTULOS = ("<3,5", "3,5–3,99", "≥4,0")


def faixa_popularidade(usuarios: pd.Series) -> pd.Series:
    """Categoriza nº de leitores em faixas fixas (NaN permanece NaN)."""
    u = pd.to_numeric(usuarios, errors="coerce")
    return pd.cut(u, bins=list(POP_CORTES), right=False, labels=list(POP_ROTULOS))


def faixa_nota(nota: pd.Series, n_avaliacoes: pd.Series,
               min_aval: int = NOTA_MIN_AVALIACOES) -> pd.Series:
    """Categoriza nota media em faixas fixas; obras com < min_aval avaliacoes ou sem nota ficam NaN."""
    r = pd.to_numeric(nota, errors="coerce")
    n = pd.to_numeric(n_avaliacoes, errors="coerce")
    r = r.where(n >= min_aval)
    return pd.cut(r, bins=list(NOTA_CORTES), right=False, labels=list(NOTA_ROTULOS))


def proporcao_por_grupo(d: pd.DataFrame, grupo: str, positivo: pd.Series, col_k: str) -> pd.DataFrame:
    """Tabela k/n/p + IC de Wilson de `positivo` por categoria de `grupo` (ordem das categorias)."""
    linhas = []
    for g, idx in d.groupby(grupo, observed=True).groups.items():
        k, n = int(positivo.loc[idx].sum()), len(idx)
        p, lo, hi = wilson_ic(k, n)
        linhas.append({"Faixa": str(g), "n": n, col_k: k, "p": p, "lo": lo, "hi": hi})
    return pd.DataFrame(linhas)


def bootstrap_ic_media(x: pd.Series, B: int = 5000, seed: int = SEED) -> Tuple[float, float]:
    """IC bootstrap percentil 95% para a media."""
    arr = x.dropna().to_numpy(dtype=float)
    if len(arr) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    medias = rng.choice(arr, size=(B, len(arr)), replace=True).mean(axis=1)
    return tuple(np.percentile(medias, [2.5, 97.5]))


def explodir_tags(tags_input: Optional[object]) -> List[str]:
    """Divide tags separadas por '|' no schema v3."""
    if pd.isna(tags_input) or not tags_input:
        return []
    return [t.strip().lower() for t in str(tags_input).split("|") if t.strip()]


def preparar_texto_wordcloud(texto: object) -> str:
    """Normaliza texto para tokenizacao (minusculas, sem URLs, sem pontuacao)."""
    if pd.isna(texto) or not texto:
        return ""
    t = str(texto).lower()
    t = re.sub(r"http\S+|www\S+", " ", t)
    t = re.sub(r"[^\w\s\-àáâãäèéêëìíîïòóôõöùúûüçñ]", " ", t)
    return t


def canonizar_editoras(df: pd.DataFrame, col: str = "Editora_PT") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Agrupa grafias da mesma editora por chave canonica, sem alterar a coluna bruta."""
    d = df.copy()
    d["Editora_Chave"] = d[col].map(chave_editora)
    com = d.dropna(subset=["Editora_Chave"])

    def melhor(s: pd.Series) -> str:
        return sorted(s.value_counts().items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))[0][0]

    rotulo = com.groupby("Editora_Chave")[col].agg(melhor)
    rotulo = rotulo.str.replace(r"^(?i:editoras?)\s+", "", regex=True)
    d["Editora_Canonica"] = d["Editora_Chave"].map(rotulo)
    aud = com.groupby("Editora_Chave")[col].agg(lambda s: sorted(set(s)))
    return d, aud[aud.map(len) > 1].to_frame("Variantes")


# =====================================================================
# CARREGAMENTO E PREPARACAO DO DATASET PARA EDA
# =====================================================================
def carregar_e_preparar_eda(caminho: Path, ano_min: int) -> Tuple[pd.DataFrame, Dict[str, int], Optional[pd.DataFrame]]:
    """Carrega CSV v3, aplica filtro de escopo e prepara colunas analiticas.
    Devolve (df_preparado, meta_filtro, auditoria_editoras_df).
    """
    log.info(f"Carregando dados de {caminho}...")
    if not caminho.exists():
        log.error(f"Arquivo nao encontrado: {caminho}")
        return pd.DataFrame(), {}, None

    df_bruto = pd.read_csv(caminho)
    log.info(f"Carregados {len(df_bruto)} registros com {len(df_bruto.columns)} colunas")

    df, meta = filtrar_escopo(df_bruto, ano_min)
    log.info(f"Filtro Ano >= {ano_min} e formato: {meta['n_inicial']} -> {meta['n_final']} edicoes "
             f"(ano < {ano_min}: {meta['n_antes_ano_min']}, sem ano: {meta['n_sem_ano']}, "
             f"formato excluido: {meta['n_formato_excluido']})")
    if df.empty:
        return df, meta, None

    aud_ed = None
    if "Editora_PT" in df.columns:
        df, aud_ed = canonizar_editoras(df, "Editora_PT")

    if "Avaliacao_Hardcover" in df.columns:
        df["Rating_Normalizado"] = pd.to_numeric(df["Avaliacao_Hardcover"], errors="coerce").round(1)
    if "Qtd_Avaliacoes_Hardcover" in df.columns:
        df["Qtd_Avaliacoes_Clean"] = pd.to_numeric(df["Qtd_Avaliacoes_Hardcover"], errors="coerce")
    if "Num_Paginas_PT" in df.columns:
        df["Num_Paginas_Clean"] = pd.to_numeric(df["Num_Paginas_PT"], errors="coerce")

    mapeamento_tags = {
        "Generos_Hardcover": "Generos",
        "Moods_Hardcover": "Moods",
        "Tags_Gerais_Hardcover": "Tags_Gerais",
        "Content_Warnings": "Content_Warnings",
    }
    for col_orig, col_dest in mapeamento_tags.items():
        if col_orig in df.columns:
            df[f"{col_dest}_list"] = df[col_orig].apply(explodir_tags)

    if "Tags_Gerais_list" in df.columns:
        df["Tags_Estruturadas_list"] = df["Tags_Gerais_list"].apply(
            lambda L: [t for t in L if t in TAGS_ESTRUTURADAS] if isinstance(L, list) else [])
        df["Tags_Gerais_list"] = df["Tags_Gerais_list"].apply(
            lambda L: [t for t in L if t not in TAGS_ESTRUTURADAS] if isinstance(L, list) else [])

    if "Titulo_PT" in df.columns:
        df["Titulo_Original_Bruto"] = df["Titulo_PT"]
        col_orig = "Titulo_Original_EN" if "Titulo_Original_EN" in df.columns else None
        df["Titulo_Limpo"] = df.apply(
            lambda r: limpar_titulo(r["Titulo_PT"], r[col_orig] if col_orig else None), axis=1)
        df["Titulo_clean"] = df["Titulo_Limpo"].apply(preparar_texto_wordcloud)

    if "Sinopse_PT" in df.columns:
        df["Sinopse_clean"] = df["Sinopse_PT"].apply(preparar_texto_wordcloud)

    return df, meta, aud_ed


# =====================================================================
# ANALISES ESTATISTICAS
# =====================================================================
def analisar_tags(ob: pd.DataFrame) -> Dict[str, Dict]:
    log.info("\n[ANALISE 1] Processando Tags (nivel obra)...")
    resultados = {}
    for cat in ("Generos", "Moods", "Tags_Gerais", "Tags_Estruturadas", "Content_Warnings"):
        col_list = f"{cat}_list"
        if col_list not in ob.columns:
            continue
        todas_tags = [t for lst in ob[col_list] if isinstance(lst, list) for t in lst]
        freq = Counter(todas_tags)
        n_com = int(ob[col_list].apply(lambda x: isinstance(x, list) and len(x) > 0).sum())
        resultados[cat] = {
            "frequencias": freq, "n_unicas": len(freq), "n_total": len(todas_tags),
            "n_registros": n_com, "cobertura_%": (n_com / len(ob)) * 100 if len(ob) else 0, "n_base": len(ob),
        }
        log.info(f"  {cat}: {len(freq)} tags unicas, {n_com}/{len(ob)} obras ({resultados[cat]['cobertura_%']:.1f}%)")
    return resultados


def _resumo(x: pd.Series) -> Dict:
    return {
        "mean": x.mean(), "median": x.median(), "std": x.std(),
        "min": x.min(), "max": x.max(),
        "q1": x.quantile(0.25), "q3": x.quantile(0.75), "count": len(x),
    }


def analisar_distribuicoes(df: pd.DataFrame) -> Dict:
    log.info("\n[ANALISE 2] Distribuicoes Continuas...")
    res = {}
    ob = deduplicar_por_obra(df)

    if "Rating_Normalizado" in ob.columns:
        r = ob["Rating_Normalizado"].dropna()
        if len(r):
            res["rating"] = _resumo(r)
            res["rating"]["ic_boot"] = bootstrap_ic_media(r)
            log.info(f"  Rating: media={res['rating']['mean']:.2f}, dp={res['rating']['std']:.2f}, n={len(r)}")

    if "Qtd_Avaliacoes_Clean" in ob.columns:
        q = ob["Qtd_Avaliacoes_Clean"].dropna()
        if len(q):
            res["qtd_avaliacoes"] = _resumo(q)

    if "Num_Paginas_Clean" in df.columns:
        p = df.loc[df["Num_Paginas_Clean"] > 0, "Num_Paginas_Clean"].dropna()
        if len(p):
            res["paginas"] = _resumo(p)

    if "Ano" in df.columns:
        a = df["Ano"].dropna()
        if len(a):
            res["ano"] = {"mean": a.mean(), "median": a.median(), "min": int(a.min()), "max": int(a.max()), "count": len(a)}
    return res


def analisar_categoricas(df: pd.DataFrame) -> Dict:
    log.info("\n[ANALISE 3] Distribuicoes Categoricas...")
    res = {}
    col_editora = "Editora_Canonica" if "Editora_Canonica" in df.columns else "Editora_PT"
    if col_editora in df.columns:
        ed = df[col_editora].dropna()
        res["editora"] = ed.value_counts().to_dict()
        log.info(f"  Editoras ({col_editora}): {len(res['editora'])} unicas, cobertura {len(ed) / len(df) * 100:.1f}%")
    if "Metodo_Ponte" in df.columns:
        res["fonte_match"] = df["Metodo_Ponte"].value_counts().to_dict()
    return res


def cobertura_por_popularidade(ob: pd.DataFrame,
                               col_users: str = "Qtd_Usuarios_Hardcover",
                               cols: Tuple[str, ...] = ("Generos_Hardcover", "Moods_Hardcover",
                                                        "Tags_Gerais_Hardcover", "Content_Warnings")
                               ) -> Optional[pd.DataFrame]:
    """Taxa de obras com >=1 tag por faixa de popularidade (nº de leitores no Hardcover)."""
    if col_users not in ob.columns:
        return None
    usuarios = pd.to_numeric(ob[col_users], errors="coerce")
    if usuarios.notna().sum() < 8:
        return None
    d = ob.copy()
    d["Pop_Faixa"] = faixa_popularidade(usuarios)
    usadas = [c for c in cols if c in d.columns]
    for c in usadas:
        d[c] = d[c].fillna("").astype(str).ne("")
    return d.groupby("Pop_Faixa", observed=True)[usadas].mean().round(3)


def sensibilidade_min_count(ob: pd.DataFrame, limiares: Tuple[int, ...] = (1, 2, 3)) -> Optional[Dict[str, Dict]]:
    """Distribuicao de Spice_Status por limiar minimo de votos por tag (requer Tags_JSON)."""
    if "Tags_JSON" not in ob.columns:
        return None
    cats = ob["Tags_JSON"].map(tags_por_categoria)
    return {f"min_count>={m}": cats.map(lambda c, m=m: spice_da_obra(c, min_count=m)["Spice_Status"])
            .value_counts().to_dict() for m in limiares}


def analisar_spice_cobertura(df: pd.DataFrame) -> Dict:
    """Conteudo sensivel / Spice x popularidade e cobertura. 'sem_sinal'/'indeterminado' = DESCONHECIDO."""
    from .rotulagem import eixos_sensiveis_obra
    from .dataset import resolver_colunas_rotulo

    ob = deduplicar_por_obra(df).copy()
    out: Dict = {}
    if "Qtd_Usuarios_Hardcover" not in ob.columns:
        return out
    u = pd.to_numeric(ob["Qtd_Usuarios_Hardcover"], errors="coerce")
    cols_tag = [c for c in ("Generos_Hardcover", "Moods_Hardcover", "Tags_Gerais_Hardcover", "Content_Warnings") if c in ob.columns]
    if cols_tag:
        n_tags = sum(ob[c].fillna("").astype(str).map(lambda s: len([t for t in s.split("|") if t.strip()])) for c in cols_tag)
        ok = u.notna()
        if ok.sum() > 5:
            try:
                from scipy.stats import spearmanr
                rho, p = spearmanr(u[ok], n_tags[ok])
            except ImportError:
                rho = float(u[ok].rank().corr(n_tags[ok].rank()))
                p = 0.0
            out["spearman_users_ntags"] = (rho, p, int(ok.sum()))
    if u.notna().sum() >= 8:
        d = ob.assign(_u=u).dropna(subset=["_u"]).copy()
        d["Pop_Faixa"] = faixa_popularidade(d["_u"])

        # Conteudo sensivel geral (qualquer um dos 4 eixos: sexo, violencia, drogas/alcool, abuso)
        cols_eixos = ("Flag_Eixo_Sexo", "Flag_Eixo_Violencia", "Flag_Eixo_Drogas_Alcool", "Flag_Eixo_Abuso_Gatilhos")
        if not all(c in d.columns for c in cols_eixos):
            eixos_df = d.apply(eixos_sensiveis_obra, axis=1, result_type="expand")
            for c in cols_eixos:
                d[c] = eixos_df[c]
        col_lbl, _, _ = resolver_colunas_rotulo(d)
        sens_pos = (d[col_lbl] == "alto") if col_lbl in d.columns else d[list(cols_eixos)].any(axis=1)
        spice_pos = (d["Spice_Status"] == "positivo") if "Spice_Status" in d.columns else None

        if spice_pos is not None:
            out["spice_por_pop"] = proporcao_por_grupo(d, "Pop_Faixa", spice_pos, "k_positivo")
        out["sensivel_por_pop"] = proporcao_por_grupo(d, "Pop_Faixa", sens_pos, "k_sensivel_alto")

        # Por faixa de NOTA (descritivo; pedido do orientador). So obras com >= NOTA_MIN_AVALIACOES votos.
        if "Avaliacao_Hardcover" in d.columns and "Qtd_Avaliacoes_Hardcover" in d.columns:
            d["Nota_Faixa"] = faixa_nota(d["Avaliacao_Hardcover"], d["Qtd_Avaliacoes_Hardcover"])
            n_aval = pd.to_numeric(d["Qtd_Avaliacoes_Hardcover"], errors="coerce")
            tem_nota = pd.to_numeric(d["Avaliacao_Hardcover"], errors="coerce").notna()
            out["nota_exclusoes"] = {
                "sem_nota": int((~tem_nota).sum()),
                f"nota_com_<{NOTA_MIN_AVALIACOES}_avaliacoes": int((tem_nota & (n_aval < NOTA_MIN_AVALIACOES)).sum()),
                "incluidas": int(d["Nota_Faixa"].notna().sum()),
            }
            dn = d.dropna(subset=["Nota_Faixa"])
            if len(dn):
                out["sensivel_por_nota"] = proporcao_por_grupo(dn, "Nota_Faixa", sens_pos.loc[dn.index], "k_sensivel_alto")
                if spice_pos is not None:
                    out["spice_por_nota"] = proporcao_por_grupo(dn, "Nota_Faixa", spice_pos.loc[dn.index], "k_positivo")
    return out


def prevalencias_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula prevalencia (com IC de Wilson 95%) de todas as flags de genero, Content Warnings
    e dos 4 eixos tematicos de conteudo sensivel (sexo, violencia, drogas/alcool, abuso/gatilhos).
    """
    from .rotulagem import eixos_sensiveis_obra

    ob = deduplicar_por_obra(df).copy()
    cols_eixos = ("Flag_Eixo_Sexo", "Flag_Eixo_Violencia", "Flag_Eixo_Drogas_Alcool", "Flag_Eixo_Abuso_Gatilhos")
    if not all(c in ob.columns for c in cols_eixos) and len(ob) > 0:
        eixos_df = ob.apply(eixos_sensiveis_obra, axis=1, result_type="expand")
        for c in cols_eixos:
            ob[c] = eixos_df[c]
    if all(c in ob.columns for c in cols_eixos):
        ob["Flag_Qualquer_Eixo_Sensivel"] = ob[list(cols_eixos)].any(axis=1)

    linhas = []
    todas_flags = (
        "Flag_Romance",
        "Flag_New_Adult",
        "Flag_Spice_Tag",
        "Flag_CW_Sexual_Content",
        "Flag_CW_Violencia",
        "Flag_CW_Drogas_Alcool",
        "Flag_CW_Abuso_Gatilhos",
        "Flag_Has_CW",
        "Flag_Closed_Door_Tag",
        "Flag_Formato_Excluir",
        "Flag_Eixo_Sexo",
        "Flag_Eixo_Violencia",
        "Flag_Eixo_Drogas_Alcool",
        "Flag_Eixo_Abuso_Gatilhos",
        "Flag_Qualquer_Eixo_Sensivel",
    )
    for c in todas_flags:
        if c in ob.columns:
            k, n = int(ob[c].astype(bool).sum()), len(ob)
            p, lo, hi = wilson_ic(k, n)
            linhas.append({"Flag": c, "k": k, "n": n, "prop": p, "IC95_inf": lo, "IC95_sup": hi})
    return pd.DataFrame(linhas)


# =====================================================================
# N-GRAMAS (DOCUMENT FREQUENCY NO NIVEL OBRA)
# =====================================================================
def tokenizar(texto: str) -> List[str]:
    return RE_TOKEN.findall(texto)


def _token_valido(t: str, extra_stop: set) -> bool:
    return len(t) >= 3 and not t.isdigit() and t not in STOP_TODAS and t not in extra_stop


def contar_ngramas(textos: List[str], extra_stop: set = frozenset()) -> Tuple[Counter, Counter]:
    """Frequencia de DOCUMENTO (cada obra conta no maximo 1x por termo) de unigramas e bigramas."""
    uni, bi = Counter(), Counter()
    for txt in textos:
        toks = tokenizar(txt)
        validos = [_token_valido(t, extra_stop) for t in toks]
        uni.update({t for t, ok in zip(toks, validos) if ok})
        bi.update({f"{a} {b}" for (a, b), (va, vb) in zip(zip(toks, toks[1:]), zip(validos, validos[1:])) if va and vb})
    return uni, bi


def tokens_de_autores(df: pd.DataFrame) -> set:
    """Tokens dos nomes de autoras (PT, Hardcover, original) para nao dominarem sinopses/titulos."""
    toks = set()
    for col in ("Autor_PT", "Autores_Hardcover", "Autor_Original_EN"):
        if col in df.columns:
            for nome in df[col].dropna().astype(str):
                for t in tokenizar(preparar_texto_wordcloud(nome)):
                    if len(t) >= 3:
                        toks.add(t)
                        toks.add(sem_acento(t))
    return toks


def calcular_ngramas_corpora(df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Tuple[Dict[str, int], Dict[str, int], int]]]:
    """Calcula tabelas e dicionarios filtrados de n-gramas por obra para titulos e sinopses."""
    ob = deduplicar_por_obra(df)
    autores = tokens_de_autores(ob)
    log.info(f"  {len(autores)} tokens de nomes de autoras removidos dos n-gramas")
    corpora = {
        "titulos": (ob["Titulo_clean"].dropna().tolist() if "Titulo_clean" in ob.columns else [],
                    autores, 1, 2, 60),
        "sinopses": (ob["Sinopse_clean"].dropna().tolist() if "Sinopse_clean" in ob.columns else [],
                     STOPWORDS_EDITORIAL | autores, 2, 2, 80),
    }
    tabelas: Dict[str, pd.DataFrame] = {}
    para_plot: Dict[str, Tuple[Dict[str, int], Dict[str, int], int]] = {}
    for nome, (textos, extra, min_df_uni, min_df_bi, max_words) in corpora.items():
        textos = [t for t in textos if t.strip()]
        if not textos:
            continue
        uni, bi = contar_ngramas(textos, extra)
        uni_f = {k: v for k, v in uni.items() if v >= min_df_uni}
        bi_f = {k: v for k, v in bi.items() if v >= min_df_bi}
        log.info(f"  {nome}: {len(textos)} textos | {len(uni_f)} unigramas (df>={min_df_uni}), "
                 f"{len(bi_f)} bigramas (df>={min_df_bi})")
        rows = [{"Corpus": nome, "Tipo": "unigrama", "Termo": k, "Freq_Doc": v} for k, v in uni.most_common(100)]
        rows += [{"Corpus": nome, "Tipo": "bigrama", "Termo": k, "Freq_Doc": v} for k, v in bi.most_common(100)]
        tabelas[nome] = pd.DataFrame(rows)
        para_plot[nome] = (uni_f, bi_f, max_words)
    return tabelas, para_plot


def bigramas_sinopse_vs_circularidade(df: pd.DataFrame, top: int = 6) -> Optional[pd.DataFrame]:
    """Top bigramas das sinopses (freq. de documento) em TODAS as obras e EXCLUINDO as que entraram no seed
    via termo de conteudo ('conteudo adulto' etc.). A diferenca mostra quanto da formula editorial e
    artefato da propria busca. Substitui a wordcloud (sem eixo, sem contagem) por algo comparavel."""
    ob = deduplicar_por_obra(df)
    if "Sinopse_clean" not in ob.columns:
        return None
    via = ob["Tipos_Origem"].fillna("").astype(str).str.contains("termo:conteudo") if "Tipos_Origem" in ob.columns \
        else pd.Series(False, index=ob.index)
    stop = STOPWORDS_EDITORIAL | tokens_de_autores(ob)
    textos_all = [t for t in ob["Sinopse_clean"].dropna() if t.strip()]
    textos_sem = [t for t in ob.loc[~via, "Sinopse_clean"].dropna() if t.strip()]
    _, bi_all = contar_ngramas(textos_all, stop)
    _, bi_sem = contar_ngramas(textos_sem, stop)
    linhas = [{"Bigrama": k, "Freq_Doc_Todas": v, "Freq_Doc_Sem_Termo_Conteudo": bi_sem.get(k, 0)}
              for k, v in bi_all.most_common(top)]
    out = pd.DataFrame(linhas)
    out.attrs["n_todas"], out.attrs["n_sem"] = len(textos_all), len(textos_sem)
    return out
