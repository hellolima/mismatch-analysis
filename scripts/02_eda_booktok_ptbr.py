"""
02_eda_booktok_ptbr.py

Pipeline de Análise Exploratória de Dados (EDA) para o dataset PT-BR consolidado (livros_ptbr_v3.csv).
Foco: engenharia de features, padronização, exploração multidimensional, figuras prontas para artigo.

Novidades desta versão (v3.2):
  - Filtro temporal: somente edições PT-BR com ano de publicação >= ANO_MIN (padrão 2015; --ano-min).
    Edições SEM ano parseável são excluídas (não dá para garantir que satisfazem o critério) e contadas no relatório.
  - Wordclouds de unigramas E bigramas (títulos e sinopses), por obra, com frequência de documento
    (cada obra conta no máximo 1x por termo), e exportação das frequências em CSV.
  - Estilo único de figuras para publicação: paleta Okabe-Ito (segura para daltônicos), fonte serif,
    larguras de 1 e 2 colunas, 300 dpi, exportação em PNG + PDF (vetorial). Títulos internos desligados por
    padrão (a legenda fica no LaTeX): ligue com --titulos.
  - Novas análises: IC de Wilson para prevalências, IC bootstrap da média do rating, distribuição de
    Score_Ponte por método, spice x popularidade (viés de cobertura), spice x tipo editorial,
    cobertura de tags por quartil (heatmap), páginas.
  - Correção: _por_obra() não colapsa mais edições sem Hardcover_Book_ID (NaN) em uma única linha.

Uso:
  python scripts/02_eda_booktok_ptbr.py [--ano-min 2015] [--titulos] [--formatos png pdf]

Outputs (outputs/): eda_report.txt, fig_*.png/.pdf, tags_frequency.csv, ngrams_*.csv,
  titulos_limpeza_auditoria.csv, editoras_normalizacao_auditoria.csv, editoras_pares_suspeitos.csv,
  cobertura_por_popularidade.csv
"""

import sys
import argparse
import warnings
import logging
import ast
import re
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from scipy.stats import spearmanr

# Utilidades compartilhadas com o seed/ponte (mesma limpar_titulo nos dois scripts)
from ptbr_common import limpar_titulo, sem_acento as _sem_acento
from editoras_norm import chave_editora, pares_suspeitos

warnings.filterwarnings('ignore')

# =====================================================================
# CONFIGURAÇÃO GERAL
# =====================================================================
BASE_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = BASE_DIR.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
CAMINHO_V3 = BASE_DIR.parent / "data" / "processed" / "livros_ptbr_v3.csv"
CAMINHO_ROTULADO = BASE_DIR.parent / "data" / "processed" / "livros_ptbr_v3_rotulado.csv"  # gerado por rotulagem.py
CAMINHO_ENTRADA = CAMINHO_ROTULADO if CAMINHO_ROTULADO.exists() else CAMINHO_V3

ANO_MIN_PADRAO = 2015
SEED = 42

# Preenchido em main(); usado no relatório
META_FILTRO: Dict[str, object] = {}

# Tags do Hardcover que parecem respostas a perguntas estruturadas de review (pares polares), não
# folksonomia livre. Separadas de Tags_Gerais para não esconder os tropos reais. Estenda se aparecerem outras.
TAGS_ESTRUTURADAS = {
    'strong character development', 'weak character development',
    'loveable characters', 'unloveable characters',
    'diverse characters', 'not diverse characters',
    'a mix driven', 'character driven', 'plot driven',
}
FIG_TITULOS = False            # títulos dentro das figuras (em artigo, use a legenda do LaTeX)
FORMATOS = ("png", "pdf")      # PNG (raster 300 dpi) + PDF (vetorial)

# =====================================================================
# ESTILO DE FIGURAS (publicação)
# =====================================================================
# Paleta Okabe-Ito: distinguível em daltonismo e em impressão P&B razoável
OKABE_ITO = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#56B4E9', '#CC79A7', '#F0E442', '#000000']
COR_PRIM = OKABE_ITO[0]     # cor principal (barras, histogramas)
COR_SEC = OKABE_ITO[1]      # destaque (média, avisos)
COR_TERC = OKABE_ITO[2]     # terciária (mediana)
COR_CINZA = '#6E6E6E'
CMAP_SEQ = 'Blues'          # heatmaps sequenciais
CMAP_WC_UNI = ['#003F66', '#0072B2', '#2E86C1', '#56B4E9']   # azuis (unigramas)
CMAP_WC_BI = ['#7A2E00', '#D55E00', '#E27D2E', '#E69F00']    # laranjas (bigramas)

W1, W2 = 3.4, 7.2           # larguras (pol.) de 1 coluna e 2 colunas (padrão A4/ABNT/SBC-like)

RC_PUB = {
    'figure.dpi': 120, 'savefig.dpi': 300,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'axes.unicode_minus': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'axes.axisbelow': True,
    'axes.prop_cycle': matplotlib.cycler(color=OKABE_ITO),
    'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,     # fontes embutidas editáveis (exigência de várias revistas)
    'figure.constrained_layout.use': True,
}


def aplicar_estilo() -> None:
    sns.set_theme(style='ticks', context='paper', rc=RC_PUB)
    plt.rcParams.update(RC_PUB)


def salvar_fig(fig: plt.Figure, nome: str) -> None:
    """Salva a figura em todos os FORMATOS com parâmetros idênticos."""
    for fmt in FORMATOS:
        fig.savefig(OUTPUT_DIR / f"{nome}.{fmt}", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    log.info(f"  Salvo: {nome}.[{'/'.join(FORMATOS)}]")


def titulo_ax(ax: plt.Axes, txt: str) -> None:
    if FIG_TITULOS:
        ax.set_title(txt, fontweight='bold')


def rotulo_painel(ax: plt.Axes, letra: str) -> None:
    """Rótulo (a), (b)... fora do eixo, canto superior esquerdo."""
    ax.text(-0.02, 1.04, f"({letra})", transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='right')


def barh_top(ax: plt.Axes, freq: Dict[str, int], n: int, cor: str, xlabel: str) -> None:
    itens = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:n][::-1]
    if not itens:
        return
    ax.barh([k for k, _ in itens], [v for _, v in itens], color=cor, edgecolor='black', linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.grid(axis='y', visible=False)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))


# =====================================================================
# LOGGING
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "eda_execution.log", encoding='utf-8'),
    ]
)
log = logging.getLogger("eda_booktok")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# =====================================================================
# UTILITÁRIOS DE LIMPEZA E PADRONIZAÇÃO
# =====================================================================

def normalizar_string(s: Optional[str]) -> str:
    if pd.isna(s) or s is None:
        return ""
    return str(s).strip().lower()


def explodir_tags(tags_input: Optional[object]) -> List[str]:
    """Explode tags em lista-string ("['a','b']"), separadas por ' | ' (v3) ou vírgula (v2)."""
    if pd.isna(tags_input) or not tags_input or tags_input == "":
        return []
    tags_str = str(tags_input).strip()
    if tags_str.startswith("[") and tags_str.endswith("]"):
        try:
            lista_eval = ast.literal_eval(tags_str)
            if isinstance(lista_eval, list):
                return [normalizar_string(t) for t in lista_eval if normalizar_string(t)]
        except (ValueError, SyntaxError):
            pass
    tags = tags_str.split("|") if "|" in tags_str else tags_str.split(",")
    return [normalizar_string(t) for t in tags if normalizar_string(t)]


def converter_ano_para_int(ano_str: Optional[object]) -> Optional[int]:
    if pd.isna(ano_str) or not ano_str:
        return None
    match = re.search(r'\d{4}', str(ano_str).strip())
    return int(match.group()) if match else None


def normalizar_rating(rating: object) -> Optional[float]:
    if pd.isna(rating) or rating == "":
        return None
    try:
        return round(float(rating), 1)
    except ValueError:
        return None


def _por_obra(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unidade de análise: rating, tags, content warnings e flags são atributos da OBRA
    (Hardcover_Book_ID); capa, editora, ano e páginas são da EDIÇÃO (ISBN).
    Deduplica por obra; edições sem ID (NaN) são mantidas individualmente (cada uma é tratada
    como obra própria), em vez de colapsar todas numa única linha.
    """
    if 'Hardcover_Book_ID' not in df.columns:
        return df
    sem_id = df[df['Hardcover_Book_ID'].isna()]
    com_id = df[df['Hardcover_Book_ID'].notna()].drop_duplicates('Hardcover_Book_ID')
    return pd.concat([com_id, sem_id]).sort_index()


def wilson_ic(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Proporção e IC de Wilson (95%). Adequado para n pequeno, ao contrário do IC de Wald."""
    if n <= 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    centro = (p + z ** 2 / (2 * n)) / den
    marg = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return p, max(0.0, centro - marg), min(1.0, centro + marg)


def bootstrap_ic_media(x: pd.Series, B: int = 5000, seed: int = SEED) -> Tuple[float, float]:
    """IC bootstrap percentil 95% para a média."""
    arr = x.dropna().to_numpy(dtype=float)
    if len(arr) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    medias = rng.choice(arr, size=(B, len(arr)), replace=True).mean(axis=1)
    return tuple(np.percentile(medias, [2.5, 97.5]))


def canonizar_editoras(df: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Agrupa grafias da mesma editora por chave canônica, sem alterar a coluna bruta."""
    d = df.copy()
    d['Editora_Chave'] = d[col].map(chave_editora)
    com = d.dropna(subset=['Editora_Chave'])

    def melhor(s: pd.Series) -> str:
        return sorted(s.value_counts().items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))[0][0]

    rotulo = com.groupby('Editora_Chave')[col].agg(melhor)
    rotulo = rotulo.str.replace(r"^(?i:editoras?)\s+", "", regex=True)
    d['Editora_Canonica'] = d['Editora_Chave'].map(rotulo)
    aud = com.groupby('Editora_Chave')[col].agg(lambda s: sorted(set(s)))
    return d, aud[aud.map(len) > 1].to_frame('Variantes')


def cobertura_por_popularidade(ob: pd.DataFrame,
                               col_users: str = 'Qtd_Usuarios_Hardcover',
                               cols: Tuple[str, ...] = ('Generos_Hardcover', 'Moods_Hardcover',
                                                        'Tags_Gerais_Hardcover', 'Content_Warnings')
                               ) -> Optional[pd.DataFrame]:
    """Taxa de obras com >=1 tag por quartil de popularidade (nº de leitores no Hardcover)."""
    if col_users not in ob.columns:
        return None
    usuarios = pd.to_numeric(ob[col_users], errors='coerce')
    if usuarios.notna().sum() < 8:
        return None
    d = ob.copy()
    d['Pop_Quartil'] = pd.qcut(usuarios.rank(method='first'), 4,
                               labels=['Q1 (menos lidas)', 'Q2', 'Q3', 'Q4 (mais lidas)'])
    usadas = [c for c in cols if c in d.columns]
    for c in usadas:
        d[c] = d[c].fillna('').astype(str).ne('')
    return d.groupby('Pop_Quartil', observed=True)[usadas].mean().round(3)


def sensibilidade_min_count(ob: pd.DataFrame, limiares: Tuple[int, ...] = (1, 2, 3)) -> Optional[Dict[str, Dict]]:
    """Distribuição de Spice_Status por limiar mínimo de votos por tag (requer Tags_JSON)."""
    if 'Tags_JSON' not in ob.columns:
        return None
    try:
        from pipeline_ptbr.labeling import tags_por_categoria, spice_da_obra
    except ImportError:
        return None
    cats = ob['Tags_JSON'].map(tags_por_categoria)
    return {f'min_count>={m}': cats.map(lambda c, m=m: spice_da_obra(c, min_count=m)['Spice_Status'])
            .value_counts().to_dict() for m in limiares}


def preparar_texto_wordcloud(texto: object) -> str:
    """Normaliza texto para tokenização (minúsculas, sem URLs, sem pontuação)."""
    if pd.isna(texto) or not texto:
        return ""
    texto = str(texto).lower()
    texto = re.sub(r'http\S+|www\S+', ' ', texto)
    texto = re.sub(r'[^\w\s\-àáâãäèéêëìíîïòóôõöùúûüçñ]', ' ', texto)
    return texto


# =====================================================================
# STOPWORDS
# =====================================================================
STOPWORDS_PT = {
    'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas',
    'de', 'em', 'para', 'com', 'por', 'sobre', 'até', 'desde', 'entre', 'durante',
    'sem', 'perante', 'mediante', 'segundo', 'conforme', 'consoante',
    'e', 'ou', 'mas', 'porém', 'contudo', 'todavia', 'entretanto', 'portanto', 'logo',
    'pois', 'que', 'se', 'como',
    'ser', 'estar', 'ter', 'haver', 'dever', 'poder', 'ir', 'vir',
    'sou', 'és', 'é', 'somos', 'sois', 'são',
    'estou', 'estás', 'está', 'estamos', 'estais', 'estão',
    'tenho', 'tens', 'tem', 'temos', 'tendes', 'têm',
    'eu', 'tu', 'ele', 'ela', 'nós', 'vós', 'eles', 'elas',
    'me', 'te', 'nos', 'vos', 'meu', 'teu', 'seu', 'nosso', 'vosso',
    'minha', 'tua', 'sua', 'nossa', 'vossa', 'meus', 'teus', 'seus', 'nossos', 'vossos',
    'minhas', 'tuas', 'suas', 'nossas', 'vossas',
    'este', 'esse', 'aquele', 'esta', 'essa', 'aquela',
    'isto', 'isso', 'aquilo', 'estes', 'esses', 'aqueles',
    'da', 'do', 'das', 'dos', 'vai', 'vão', 'será', 'serão',
    'foi', 'fomos', 'foram', 'era', 'éramos', 'eram',
    'muito', 'pouco', 'todo', 'nenhum', 'algum', 'outro', 'mesmo', 'próprio',
    'mais', 'menos', 'bem', 'mal', 'cá', 'aí', 'lá',
    'já', 'ainda', 'sempre', 'nunca', 'jamais', 'talvez', 'também', 'nem', 'sequer',
    'aqui', 'agora', 'hoje', 'ontem', 'amanhã', 'quando', 'onde', 'porque', 'quem',
}

STOPWORDS_PT_EXTRA = {
    'no', 'na', 'nas', 'ao', 'aos', 'à', 'às', 'pelo', 'pela', 'pelos', 'pelas',
    'num', 'numa', 'nuns', 'numas', 'dum', 'duma', 'duns', 'dumas',
    'dele', 'dela', 'deles', 'delas', 'neste', 'nesta', 'nestes', 'nestas',
    'nesse', 'nessa', 'nesses', 'nessas', 'naquele', 'naquela', 'naqueles', 'naquelas',
    'nisto', 'nisso', 'naquilo', 'deste', 'desta', 'destes', 'destas',
    'desse', 'dessa', 'desses', 'dessas', 'daquele', 'daquela', 'daqueles', 'daquelas',
    'disto', 'disso', 'daquilo', 'àquele', 'àquela', 'àqueles', 'àquelas',
    'lhe', 'lhes', 'mim', 'ti', 'si', 'consigo', 'comigo', 'contigo', 'conosco', 'convosco',
    'você', 'vocês', 'outra', 'outros', 'outras', 'tal', 'tais', 'qual', 'quais', 'quê',
    'cujo', 'cuja', 'cujos', 'cujas', 'quanto', 'quanta', 'quantos', 'quantas',
    'alguém', 'ninguém', 'tudo', 'nada', 'algo', 'cada', 'vários', 'várias', 'alguns', 'algumas',
    'nenhuma', 'nenhuns', 'nenhumas', 'toda', 'todos', 'todas', 'certo', 'certa',
    'não', 'sim', 'assim', 'apenas', 'só', 'somente', 'depois', 'antes', 'após', 'então',
    'enquanto', 'embora', 'apesar', 'entanto', 'tanto', 'tão', 'quase', 'tampouco',
    'ali', 'além', 'através', 'sob', 'contra', 'ante',
    'inclusive', 'sendo', 'caso', 'dado',
    'pra', 'pro', 'pras', 'pros', 'né', 'tá',
    'dois', 'duas', 'três',
    'sido', 'fui', 'fora', 'seja', 'sejam', 'seria', 'serem',
    'estava', 'estavam', 'esteja', 'estive', 'estiver',
    'tinha', 'tinham', 'tenha', 'tenham', 'tido', 'tendo', 'teve', 'tiver',
    'há', 'havia', 'houve', 'haja',
    'pode', 'podem', 'podia', 'podiam', 'possa', 'possam', 'pôde',
    'deve', 'devem', 'devia', 'vem', 'vêm', 'veio', 'vindo', 'vamos', 'ia', 'iam',
    'quer', 'querem', 'queria', 'queriam', 'quis',
}
STOPWORDS_PT = STOPWORDS_PT | STOPWORDS_PT_EXTRA
STOPWORDS_PT = STOPWORDS_PT | {_sem_acento(w) for w in STOPWORDS_PT}

# Edições brasileiras que mantêm o título em inglês ("The Risk"); sem isto "the"/"of" dominam.
STOPWORDS_EN = {
    'the', 'of', 'and', 'to', 'in', 'on', 'at', 'for', 'with', 'a', 'an',
    'is', 'it', 'my', 'your', 'you', 'we', 'our', 'his', 'her', 'its',
}

# Ruído editorial recorrente em sinopses (não temático). Ajuste conforme inspecionar ngrams_*.csv.
STOPWORDS_EDITORIAL = {
    'livro', 'livros', 'autora', 'autor', 'romance', 'romances', 'best-seller', 'bestseller',
    'sucesso', 'leitores', 'leitor', 'leitura', 'edição', 'edicao', 'série', 'serie', 'volume',
    'história', 'historia', 'ler', 'new', 'york', 'times', 'tiktok', 'booktok',
    'publishers', 'weekly', 'kirkus', 'reviews', 'goodreads', 'amazon',
}
STOPWORDS_EDITORIAL |= {_sem_acento(w) for w in STOPWORDS_EDITORIAL}

STOP_TODAS = STOPWORDS_PT | STOPWORDS_EN
RE_TOKEN = re.compile(r"[a-zà-ÿ0-9]+(?:-[a-zà-ÿ0-9]+)*")


# =====================================================================
# N-GRAMAS
# =====================================================================

def tokenizar(texto: str) -> List[str]:
    return RE_TOKEN.findall(texto)


def _token_valido(t: str, extra_stop: set) -> bool:
    return len(t) >= 3 and not t.isdigit() and t not in STOP_TODAS and t not in extra_stop


def contar_ngramas(textos: List[str], extra_stop: set = frozenset()) -> Tuple[Counter, Counter]:
    """
    Frequência de DOCUMENTO (cada texto conta no máximo 1x por termo) de unigramas e bigramas.
    Bigramas: pares de tokens ADJACENTES no texto original, ambos válidos. Stopwords agem como
    barreira: 'rei da noite' não gera 'rei noite'. Frequência de documento evita que uma única
    sinopse repetitiva domine a nuvem.
    """
    uni, bi = Counter(), Counter()
    for txt in textos:
        toks = tokenizar(txt)
        validos = [_token_valido(t, extra_stop) for t in toks]
        uni.update({t for t, ok in zip(toks, validos) if ok})
        bi.update({f"{a} {b}" for (a, b), (va, vb) in zip(zip(toks, toks[1:]), zip(validos, validos[1:])) if va and vb})
    return uni, bi


def _cor_wc(paleta: List[str]):
    def fn(word=None, font_size=None, position=None, orientation=None, random_state=None, **kw):
        rng = np.random.default_rng(abs(hash(word)) % (2 ** 32))
        return paleta[int(rng.integers(0, len(paleta)))]
    return fn


def _wc_img(freq: Dict[str, int], paleta: List[str], max_words: int, w: int = 1200, h: int = 800):
    if not freq:
        return None
    return WordCloud(width=w, height=h, background_color='white', color_func=_cor_wc(paleta),
                     max_words=max_words, prefer_horizontal=0.9, relative_scaling=0.6,
                     min_font_size=8, random_state=SEED, collocations=False,
                     font_path=None).generate_from_frequencies(dict(freq))


def tokens_de_autores(df: pd.DataFrame) -> set:
    """Tokens dos nomes de autoras (PT, Hardcover, original) para não dominarem sinopses/títulos
    (ex.: 'emily henry' aparece em blurbs). Versões com e sem acento."""
    toks = set()
    for col in ('Autor_PT', 'Autores_Hardcover', 'Autor_Original_EN'):
        if col in df.columns:
            for nome in df[col].dropna().astype(str):
                for t in tokenizar(preparar_texto_wordcloud(nome)):
                    if len(t) >= 3:
                        toks.add(t)
                        toks.add(_sem_acento(t))
    return toks


def gerar_wordclouds(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Wordclouds de unigramas e bigramas (títulos e sinopses), nível OBRA; CSV de frequências."""
    log.info("\n[ANÁLISE 4] Wordclouds (unigramas + bigramas)...")
    ob = _por_obra(df)
    autores = tokens_de_autores(ob)
    log.info(f"  {len(autores)} tokens de nomes de autoras removidos dos n-gramas")
    # (textos, stopwords extras, min_df unigramas, min_df bigramas, max_words)
    # Títulos: bigramas com df=1 só listam títulos individuais, por isso min_df=2 nos bigramas.
    corpora = {
        'titulos': (ob['Titulo_clean'].dropna().tolist() if 'Titulo_clean' in ob.columns else [],
                    autores, 1, 2, 60),
        'sinopses': (ob['Sinopse_clean'].dropna().tolist() if 'Sinopse_clean' in ob.columns else [],
                     STOPWORDS_EDITORIAL | autores, 2, 2, 80),
    }
    tabelas = {}
    for nome, (textos, extra, min_df_uni, min_df_bi, max_words) in corpora.items():
        textos = [t for t in textos if t.strip()]
        if not textos:
            continue
        uni, bi = contar_ngramas(textos, extra)
        uni_f = {k: v for k, v in uni.items() if v >= min_df_uni}
        bi_f = {k: v for k, v in bi.items() if v >= min_df_bi}
        log.info(f"  {nome}: {len(textos)} textos | {len(uni_f)} unigramas (df>={min_df_uni}), "
                 f"{len(bi_f)} bigramas (df>={min_df_bi})")

        rows = [{'Corpus': nome, 'Tipo': 'unigrama', 'Termo': k, 'Freq_Doc': v} for k, v in uni.most_common(100)]
        rows += [{'Corpus': nome, 'Tipo': 'bigrama', 'Termo': k, 'Freq_Doc': v} for k, v in bi.most_common(100)]
        tabelas[nome] = pd.DataFrame(rows)
        tabelas[nome].to_csv(OUTPUT_DIR / f'ngrams_{nome}.csv', index=False, encoding='utf-8')

        fig, axes = plt.subplots(1, 2, figsize=(W2, 3.2))
        for ax, (freq, paleta, letra, rot) in zip(
                axes, [(uni_f, CMAP_WC_UNI, 'a', 'Palavras (unigramas)'),
                       (bi_f, CMAP_WC_BI, 'b', 'Bigramas')]):
            wc = _wc_img(freq, paleta, max_words)
            ax.axis('off')
            if wc is not None:
                ax.imshow(wc, interpolation='bilinear')
            else:
                ax.text(0.5, 0.5, 'sem termos acima do limiar', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"({letra}) {rot}", fontsize=9, loc='left')
        if FIG_TITULOS:
            fig.suptitle(f"Nuvens de termos: {nome} (frequência de documento, nível obra)", fontweight='bold')
        salvar_fig(fig, f'fig_wordcloud_{nome}')
    return tabelas


# =====================================================================
# CARREGAMENTO E LIMPEZA DE DADOS
# =====================================================================

def carregar_e_limpar_dados(caminho: Path, ano_min: int) -> pd.DataFrame:
    log.info(f"Carregando dados de {caminho}...")
    if not caminho.exists():
        log.error(f"Arquivo não encontrado: {caminho}")
        return pd.DataFrame()

    df = pd.read_csv(caminho)
    n_bruto_total = len(df)
    log.info(f"Carregados {n_bruto_total} registros com {len(df.columns)} colunas")

    # ---- Ano da edição PT-BR + FILTRO TEMPORAL (antes de qualquer estatística)
    col_ano = next((c for c in ('Ano_Publicacao_PT', 'Ano_Publicacao') if c in df.columns), None)
    if not col_ano:
        log.error("Coluna de ano PT não encontrada: impossível aplicar o filtro temporal. Abortando.")
        return pd.DataFrame()
    df['Ano'] = df[col_ano].apply(converter_ano_para_int)
    n_sem_ano = int(df['Ano'].isna().sum())
    n_antes = int((df['Ano'] < ano_min).sum())
    df = df[df['Ano'] >= ano_min].copy().reset_index(drop=True)
    META_FILTRO.update({'ano_min': ano_min, 'n_inicial': n_bruto_total, 'n_sem_ano': n_sem_ano,
                        'n_antes_ano_min': n_antes, 'n_final': len(df)})
    log.info(f"Filtro Ano >= {ano_min}: {n_bruto_total} -> {len(df)} edições "
             f"(removidas: {n_antes} anteriores a {ano_min}, {n_sem_ano} sem ano parseável)")
    if df.empty:
        return df

    # ---- Exclusão de formatos fora de escopo (mangá, quadrinho, graphic novel, infantil)
    if 'Flag_Formato_Excluir' in df.columns:
        excl = df['Flag_Formato_Excluir'].astype(str).str.lower().isin(['true', '1', '1.0'])
        META_FILTRO['n_formato_excluido'] = int(excl.sum())
        df = df[~excl].copy().reset_index(drop=True)
        META_FILTRO['n_final'] = len(df)
        log.info(f"Formato fora de escopo excluído: {META_FILTRO['n_formato_excluido']} edições -> {len(df)} restantes")
    if df.empty:
        return df

    # Editoras: chave canônica
    col_ed_raw = next((c for c in ('Editora_PT', 'Editora') if c in df.columns), None)
    if col_ed_raw:
        n_bruto = df[col_ed_raw].dropna().nunique()
        df, aud_ed = canonizar_editoras(df, col_ed_raw)
        aud_ed.to_csv(OUTPUT_DIR / 'editoras_normalizacao_auditoria.csv', encoding='utf-8')
        n_canon = df['Editora_Chave'].dropna().nunique()
        log.info(f"Editoras: {n_bruto} grafias brutas -> {n_canon} canônicas ({len(aud_ed)} grupos fundiram 2+ grafias)")
        sus = pares_suspeitos(df['Editora_Chave'].dropna().unique())
        pd.DataFrame(sus, columns=['Chave_Curta', 'Chave_Longa']).to_csv(
            OUTPUT_DIR / 'editoras_pares_suspeitos.csv', index=False, encoding='utf-8')
        if sus:
            log.info(f"Editoras: {len(sus)} pares suspeitos em editoras_pares_suspeitos.csv; registre em ALIASES_EDITORA")

    if 'Rating' in df.columns:
        df['Rating_Normalizado'] = df['Rating'].apply(normalizar_rating)
    elif 'Avaliacao_Hardcover' in df.columns:
        df['Rating_Normalizado'] = df['Avaliacao_Hardcover'].apply(normalizar_rating)

    if 'Qtd_Avaliacoes' in df.columns:
        df['Qtd_Avaliacoes_Clean'] = pd.to_numeric(df['Qtd_Avaliacoes'], errors='coerce')
    elif 'Qtd_Avaliacoes_Hardcover' in df.columns:
        df['Qtd_Avaliacoes_Clean'] = pd.to_numeric(df['Qtd_Avaliacoes_Hardcover'], errors='coerce')

    if 'Paginas' in df.columns:
        df['Num_Paginas_Clean'] = pd.to_numeric(df['Paginas'], errors='coerce')
    elif 'Num_Paginas_PT' in df.columns:
        df['Num_Paginas_Clean'] = pd.to_numeric(df['Num_Paginas_PT'], errors='coerce')

    mapeamento_tags = {
        'Generos': 'Generos', 'Generos_Hardcover': 'Generos',
        'Moods': 'Moods', 'Moods_Hardcover': 'Moods',
        'Tags_Gerais': 'Tags_Gerais', 'Tags_Gerais_Hardcover': 'Tags_Gerais',
        'Content_Warnings': 'Content_Warnings',
    }
    for col_orig, col_destino in mapeamento_tags.items():
        if col_orig in df.columns:
            df[f'{col_destino}_list'] = df[col_orig].apply(explodir_tags)

    # Tags_Gerais = só tropos/folksonomia livre; as respostas estruturadas vão para Tags_Estruturadas
    if 'Tags_Gerais_list' in df.columns:
        df['Tags_Estruturadas_list'] = df['Tags_Gerais_list'].apply(
            lambda L: [t for t in L if t in TAGS_ESTRUTURADAS] if isinstance(L, list) else [])
        df['Tags_Gerais_list'] = df['Tags_Gerais_list'].apply(
            lambda L: [t for t in L if t not in TAGS_ESTRUTURADAS] if isinstance(L, list) else [])

    col_titulo = 'Titulo' if 'Titulo' in df.columns else 'Titulo_PT'
    if col_titulo in df.columns:
        df['Titulo_Original_Bruto'] = df[col_titulo]
        col_orig = 'Titulo_Original_EN' if 'Titulo_Original_EN' in df.columns else None
        df['Titulo_Limpo'] = df.apply(
            lambda r: limpar_titulo(r[col_titulo], r[col_orig] if col_orig else None), axis=1)
        df['Titulo_clean'] = df['Titulo_Limpo'].apply(preparar_texto_wordcloud)

    col_sinopse = 'Sinopse' if 'Sinopse' in df.columns else 'Sinopse_PT'
    if col_sinopse in df.columns:
        df['Sinopse_clean'] = df[col_sinopse].apply(preparar_texto_wordcloud)

    return df


# =====================================================================
# ANÁLISES ESTATÍSTICAS
# =====================================================================

def analisar_tags(df: pd.DataFrame) -> Dict[str, Dict]:
    log.info("\n[ANÁLISE 1] Processando Tags...")
    resultados = {}
    for cat in ['Generos', 'Moods', 'Tags_Gerais', 'Tags_Estruturadas', 'Content_Warnings']:
        col_list = f'{cat}_list'
        if col_list not in df.columns:
            continue
        todas_tags = [t for lst in df[col_list] if isinstance(lst, list) for t in lst]
        freq = Counter(todas_tags)
        n_com = int(df[col_list].apply(lambda x: isinstance(x, list) and len(x) > 0).sum())
        resultados[cat] = {
            'frequencias': freq, 'n_unicas': len(freq), 'n_total': len(todas_tags),
            'n_registros': n_com, 'cobertura_%': (n_com / len(df)) * 100 if len(df) else 0, 'n_base': len(df),
        }
        log.info(f"  {cat}: {len(freq)} tags únicas, {n_com}/{len(df)} obras ({resultados[cat]['cobertura_%']:.1f}%)")
    return resultados


def _resumo(x: pd.Series) -> Dict:
    return {'mean': x.mean(), 'median': x.median(), 'std': x.std(), 'min': x.min(), 'max': x.max(),
            'q1': x.quantile(.25), 'q3': x.quantile(.75), 'count': len(x)}


def analisar_distribuicoes(df: pd.DataFrame) -> Dict:
    log.info("\n[ANÁLISE 2] Distribuições Contínuas...")
    res = {}
    ob = _por_obra(df)

    if 'Rating_Normalizado' in df.columns:
        r = ob['Rating_Normalizado'].dropna()
        if len(r):
            res['rating'] = _resumo(r)
            res['rating']['ic_boot'] = bootstrap_ic_media(r)
            log.info(f"  Rating: média={res['rating']['mean']:.2f}, dp={res['rating']['std']:.2f}, n={len(r)}")

    if 'Qtd_Avaliacoes_Clean' in df.columns:
        q = ob['Qtd_Avaliacoes_Clean'].dropna()
        if len(q):
            res['qtd_avaliacoes'] = _resumo(q)

    if 'Num_Paginas_Clean' in df.columns:
        p = df.loc[df['Num_Paginas_Clean'] > 0, 'Num_Paginas_Clean'].dropna()
        if len(p):
            res['paginas'] = _resumo(p)

    if 'Ano' in df.columns:
        a = df['Ano'].dropna()
        if len(a):
            res['ano'] = {'mean': a.mean(), 'median': a.median(), 'min': int(a.min()), 'max': int(a.max()), 'count': len(a)}
    return res


def analisar_categoricas(df: pd.DataFrame) -> Dict:
    log.info("\n[ANÁLISE 3] Distribuições Categóricas...")
    res = {}
    col_editora = next((c for c in ('Editora_Canonica', 'Editora', 'Editora_PT') if c in df.columns), None)
    if col_editora:
        ed = df[col_editora].dropna()
        res['editora'] = ed.value_counts().to_dict()
        log.info(f"  Editoras ({col_editora}): {len(res['editora'])} únicas, cobertura {len(ed) / len(df) * 100:.1f}%")
    col_fonte = next((c for c in ('Metodo_Ponte', 'Fonte_Match') if c in df.columns), None)
    if col_fonte:
        res['fonte_match'] = df[col_fonte].value_counts().to_dict()
    return res


def analisar_spice_cobertura(df: pd.DataFrame) -> Dict:
    """
    Spice x popularidade e cobertura. 'sem_sinal' = DESCONHECIDO. Se a taxa de spice positivo cresce
    com a popularidade, ela reflete em parte a densidade de tags, não só o conteúdo (viés de medição).
    """
    ob = _por_obra(df)
    out: Dict = {}
    if 'Qtd_Usuarios_Hardcover' not in ob.columns:
        return out
    u = pd.to_numeric(ob['Qtd_Usuarios_Hardcover'], errors='coerce')
    cols_tag = [c for c in ('Generos_Hardcover', 'Moods_Hardcover', 'Tags_Gerais_Hardcover', 'Content_Warnings') if c in ob.columns]
    if cols_tag:
        n_tags = sum(ob[c].fillna('').astype(str).map(lambda s: len([t for t in s.split('|') if t.strip()])) for c in cols_tag)
        ok = u.notna()
        if ok.sum() > 5:
            rho, p = spearmanr(u[ok], n_tags[ok])
            out['spearman_users_ntags'] = (rho, p, int(ok.sum()))
    if 'Spice_Status' in ob.columns and u.notna().sum() >= 8:
        d = ob.assign(_u=u).dropna(subset=['_u'])
        d['Pop_Quartil'] = pd.qcut(d['_u'].rank(method='first'), 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        linhas = []
        for q, g in d.groupby('Pop_Quartil', observed=True):
            k, n = int((g['Spice_Status'] == 'positivo').sum()), len(g)
            p, lo, hi = wilson_ic(k, n)
            linhas.append({'Quartil': q, 'n': n, 'k_positivo': k, 'p': p, 'lo': lo, 'hi': hi})
        out['spice_por_pop'] = pd.DataFrame(linhas)
    return out


def prevalencias_flags(df: pd.DataFrame) -> pd.DataFrame:
    ob = _por_obra(df)
    linhas = []
    for c in ('Flag_Romance', 'Flag_New_Adult', 'Flag_Spice_Tag', 'Flag_CW_Sexual_Content',
              'Flag_Closed_Door_Tag', 'Flag_Formato_Excluir'):
        if c in ob.columns:
            k, n = int(ob[c].astype(bool).sum()), len(ob)
            p, lo, hi = wilson_ic(k, n)
            linhas.append({'Flag': c, 'k': k, 'n': n, 'prop': p, 'IC95_inf': lo, 'IC95_sup': hi})
    return pd.DataFrame(linhas)


# =====================================================================
# VISUALIZAÇÕES
# =====================================================================

def gerar_visualizacoes(df: pd.DataFrame, stats_tags: Dict, stats_dist: Dict, stats_cat: Dict,
                        extra: Dict) -> None:
    log.info("\n[ANÁLISE 5] Gerando Visualizações...")
    ob = _por_obra(df)

    # FIG 1: Ratings (histograma + boxplot horizontal compartilhando o eixo)
    if 'Rating_Normalizado' in df.columns:
        r = ob['Rating_Normalizado'].dropna()
        if len(r):
            fig, (ax, axb) = plt.subplots(2, 1, figsize=(W1 + 1.0, 3.2), sharex=True,
                                          gridspec_kw={'height_ratios': [4, 1]})
            ax.hist(r, bins=np.arange(1.0, 5.25, 0.25), color=COR_PRIM, edgecolor='black', linewidth=0.5, alpha=0.9)
            ax.axvline(r.mean(), color=COR_SEC, ls='--', lw=1.3, label=f'Média = {r.mean():.2f}')
            ax.axvline(r.median(), color=COR_TERC, ls=':', lw=1.5, label=f'Mediana = {r.median():.2f}')
            ax.set_ylabel('Nº de obras')
            ax.legend(loc='upper left')
            titulo_ax(ax, 'Distribuição de ratings (nível obra)')
            axb.boxplot(r, vert=False, widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor=COR_PRIM, alpha=0.5), medianprops=dict(color='black'))
            axb.set_yticks([])
            axb.set_xlabel('Rating (Hardcover, 1–5)')
            axb.grid(axis='y', visible=False)
            salvar_fig(fig, 'fig_01_ratings_dist')

    # FIG 2: Distribuição temporal (edições PT-BR por ano; eixo contínuo, anos sem edição = 0)
    if 'Ano' in df.columns and df['Ano'].notna().any():
        a = df['Ano'].dropna().astype(int)
        idx = range(int(a.min()), int(a.max()) + 1)
        cont = a.value_counts().reindex(idx, fill_value=0)
        fig, ax = plt.subplots(figsize=(W2, 2.6))
        ax.bar(cont.index.astype(str), cont.values, color=COR_PRIM, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Ano de publicação da edição PT-BR')
        ax.set_ylabel('Nº de edições')
        ax.grid(axis='x', visible=False)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        titulo_ax(ax, 'Distribuição temporal das edições PT-BR')
        salvar_fig(fig, 'fig_02_anos_dist')

    # FIG 3: Top 10 editoras
    if stats_cat.get('editora'):
        fig, ax = plt.subplots(figsize=(W1 + 1.0, 3.0))
        barh_top(ax, stats_cat['editora'], 10, COR_PRIM, 'Nº de edições')
        titulo_ax(ax, 'Top 10 editoras (grafias agrupadas)')
        salvar_fig(fig, 'fig_03_editoras_top10')

    # FIG 4: Top tags (painéis empilhados, 1 coluna larga)
    cats = [c for c in ('Generos', 'Moods', 'Tags_Gerais') if c in stats_tags]
    if cats:
        fig, axes = plt.subplots(len(cats), 1, figsize=(W1 + 1.0, 2.7 * len(cats)))
        axes = np.atleast_1d(axes)
        nomes = {'Generos': 'Gêneros', 'Moods': 'Moods', 'Tags_Gerais': 'Tags gerais'}
        for i, (ax, c) in enumerate(zip(axes, cats)):
            barh_top(ax, stats_tags[c]['frequencias'], 12, OKABE_ITO[i], 'Nº de obras')
            ax.set_title(f"({'abc'[i]}) {nomes[c]}", loc='left', fontsize=9)
        salvar_fig(fig, 'fig_04_tags_top')

    # FIG 5: Rating vs nº de avaliações (escala log; Spearman)
    if {'Rating_Normalizado', 'Qtd_Avaliacoes_Clean'} <= set(df.columns):
        d = ob.dropna(subset=['Rating_Normalizado', 'Qtd_Avaliacoes_Clean'])
        d = d[d['Qtd_Avaliacoes_Clean'] > 0]
        if len(d) > 2:
            fig, ax = plt.subplots(figsize=(W1 + 1.0, 3.0))
            ax.scatter(d['Qtd_Avaliacoes_Clean'], d['Rating_Normalizado'], s=18, alpha=0.7,
                       color=COR_PRIM, edgecolors='black', linewidths=0.4)
            ax.set_xscale('log')
            rho, p = spearmanr(d['Qtd_Avaliacoes_Clean'], d['Rating_Normalizado'])
            ptxt = 'p < 0,001' if p < 0.001 else f'p = {p:.3f}'.replace('.', ',')
            ax.text(0.04, 0.05, f'Spearman ρ = {rho:.2f}'.replace('.', ',') + f' ({ptxt}; n = {len(d)})',
                    transform=ax.transAxes, fontsize=8, bbox=dict(boxstyle='round', fc='white', ec=COR_CINZA, lw=0.5))
            ax.set_xlabel('Nº de avaliações (escala log)')
            ax.set_ylabel('Rating')
            titulo_ax(ax, 'Rating × nº de avaliações')
            salvar_fig(fig, 'fig_05_rating_vs_qtd_scatter')

    # FIG 6: Content warnings (cor sequencial única; intensidade = frequência)
    cw = stats_tags.get('Content_Warnings', {}).get('frequencias')
    if cw:
        fig, ax = plt.subplots(figsize=(W1 + 1.0, 3.6))
        barh_top(ax, cw, 15, COR_SEC, 'Nº de obras')
        titulo_ax(ax, 'Top 15 content warnings')
        salvar_fig(fig, 'fig_06_content_warnings')

    # FIG 7: Método de resolução da ponte (barras, não pizza: comparação de comprimento > ângulo)
    if stats_cat.get('fonte_match'):
        fm = pd.Series(stats_cat['fonte_match']).sort_values()
        fig, ax = plt.subplots(figsize=(W1 + 1.0, 2.0))
        ax.barh(fm.index, fm.values, color=COR_PRIM, edgecolor='black', linewidth=0.5)
        for y, v in enumerate(fm.values):
            ax.text(v, y, f' {v} ({100 * v / fm.sum():.1f}%)'.replace('.', ','), va='center', fontsize=8)
        ax.set_xlim(0, fm.max() * 1.3)
        ax.set_xlabel('Nº de edições')
        ax.grid(axis='y', visible=False)
        titulo_ax(ax, 'Método de resolução da ponte')
        salvar_fig(fig, 'fig_07_fonte_match')

    # FIG 8: Score_Ponte por método
    if {'Score_Ponte', 'Metodo_Ponte'} <= set(df.columns):
        d = df.dropna(subset=['Score_Ponte', 'Metodo_Ponte'])
        if len(d):
            ordem = d.groupby('Metodo_Ponte')['Score_Ponte'].median().sort_values().index.tolist()
            fig, ax = plt.subplots(figsize=(W1 + 1.0, 2.4))
            sns.boxplot(data=d, y='Metodo_Ponte', x='Score_Ponte', order=ordem, color=COR_PRIM, width=0.5,
                        fliersize=0, boxprops=dict(alpha=0.4), ax=ax)
            sns.stripplot(data=d, y='Metodo_Ponte', x='Score_Ponte', order=ordem, color='black', size=2.5,
                          alpha=0.6, jitter=0.2, ax=ax)
            ax.set_xlabel('Score da ponte')
            ax.set_ylabel('')
            ax.set_xlim(0, 1.02)
            titulo_ax(ax, 'Score da ponte por método')
            salvar_fig(fig, 'fig_08_score_ponte')

    # FIG 9: Páginas
    if stats_dist.get('paginas'):
        p = df.loc[df['Num_Paginas_Clean'] > 0, 'Num_Paginas_Clean'].dropna()
        fig, ax = plt.subplots(figsize=(W1 + 1.0, 2.5))
        ax.hist(p, bins=20, color=COR_PRIM, edgecolor='black', linewidth=0.5)
        ax.axvline(p.median(), color=COR_SEC, ls='--', lw=1.3, label=f'Mediana = {p.median():.0f}')
        ax.set_xlabel('Nº de páginas (edição PT-BR)')
        ax.set_ylabel('Nº de edições')
        ax.legend()
        titulo_ax(ax, 'Distribuição de páginas')
        salvar_fig(fig, 'fig_09_paginas')

    # FIG 10: Spice_Status / Spice_Nivel (obra); 'sem_sinal' em cinza: é desconhecido, não ausência
    if 'Spice_Status' in ob.columns:
        fig, axes = plt.subplots(1, 2, figsize=(W2, 2.6))
        vc = ob['Spice_Status'].value_counts()
        cores = [COR_CINZA if k == 'sem_sinal' else COR_SEC for k in vc.index]
        axes[0].bar(vc.index, vc.values, color=cores, edgecolor='black', linewidth=0.5)
        for i, v in enumerate(vc.values):
            axes[0].text(i, v, f'{v}\n({100 * v / vc.sum():.1f}%)'.replace('.', ','), ha='center', va='bottom', fontsize=8)
        axes[0].set_ylim(0, vc.max() * 1.25)
        axes[0].set_ylabel('Nº de obras')
        axes[0].set_title('(a) Spice_Status', loc='left', fontsize=9)
        axes[0].grid(axis='x', visible=False)
        if 'Spice_Nivel' in ob.columns:
            niv = pd.to_numeric(ob['Spice_Nivel'], errors='coerce').dropna().astype(int).value_counts().reindex([0, 1, 2, 3], fill_value=0)
            axes[1].bar(['0\nfechado', '1\nsensual', '2\ncena aberta', '3\nexplícito'], niv.values,
                        color=COR_SEC, edgecolor='black', linewidth=0.5)
            axes[1].set_ylabel('Nº de obras (com sinal)')
            axes[1].set_title('(b) Spice_Nivel', loc='left', fontsize=9)
            axes[1].grid(axis='x', visible=False)
        salvar_fig(fig, 'fig_10_spice')

    # FIG 11: Cobertura de tags por quartil de popularidade (heatmap)
    cob = cobertura_por_popularidade(ob)
    if cob is not None:
        fig, ax = plt.subplots(figsize=(W1 + 1.4, 2.6))
        c2 = cob.copy()
        c2.columns = [c.replace('_Hardcover', '').replace('_', ' ') for c in c2.columns]
        sns.heatmap(c2, annot=True, fmt='.2f', cmap=CMAP_SEQ, vmin=0, vmax=1, linewidths=0.5, linecolor='white',
                    cbar_kws={'label': 'Fração de obras com ≥1 tag'}, ax=ax)
        ax.set_ylabel('Quartil de popularidade')
        ax.set_xlabel('')
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        salvar_fig(fig, 'fig_11_cobertura_popularidade')

    # FIG 12: Spice positivo por quartil de popularidade (IC de Wilson)
    sp = extra.get('spice_por_pop')
    if sp is not None and len(sp):
        fig, ax = plt.subplots(figsize=(W1 + 0.6, 2.6))
        ax.errorbar(sp['Quartil'], sp['p'], yerr=[sp['p'] - sp['lo'], sp['hi'] - sp['p']], fmt='o', color=COR_SEC,
                    ecolor='black', capsize=3, markersize=5)
        for x, (p, n) in enumerate(zip(sp['p'], sp['n'])):
            ax.text(x + 0.08, p, f'n={n}', fontsize=7, va='center')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Prop. de obras com spice positivo')
        ax.set_xlabel('Quartil de popularidade (nº de leitores)')
        titulo_ax(ax, 'Spice positivo × popularidade (IC de Wilson 95%)')
        salvar_fig(fig, 'fig_12_spice_vs_popularidade')

    # FIG 13: Tipo editorial × Spice_Status (descritivo; proporção por linha, contagens anotadas)
    if {'Spice_Status', 'Tipo_Editorial_Editora'} <= set(ob.columns):
        ct = pd.crosstab(ob['Tipo_Editorial_Editora'].fillna('NA'), ob['Spice_Status'])
        prop = ct.div(ct.sum(axis=1), axis=0)
        annot = ct.astype(str) + '\n(' + (prop * 100).round(0).astype(int).astype(str) + '%)'
        fig, ax = plt.subplots(figsize=(W1 + 1.0, 2.8))
        sns.heatmap(prop, annot=annot, fmt='', cmap=CMAP_SEQ, vmin=0, vmax=1, linewidths=0.5, linecolor='white',
                    cbar_kws={'label': 'Proporção por linha'}, ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel('Tipo editorial')
        salvar_fig(fig, 'fig_13_tipo_editorial_spice')

    # FIG 14: Rótulo triangulado de spice (obra), empilhado por confiança
    if {'Spice_Label', 'Spice_Label_Confianca'} <= set(ob.columns):
        ct = pd.crosstab(ob['Spice_Label'], ob['Spice_Label_Confianca']).reindex(
            ['alto', 'baixo', 'provavel_baixo', 'indeterminado']).fillna(0)
        cores = {'forte': OKABE_ITO[0], 'fraca': OKABE_ITO[4], 'conflito': OKABE_ITO[3],
                 'nao_validado': OKABE_ITO[5], 'sem_evidencia': COR_CINZA}
        fig, ax = plt.subplots(figsize=(W1 + 1.0, 2.6))
        base = np.zeros(len(ct))
        for col in ct.columns:
            ax.bar(ct.index, ct[col].values, bottom=base, label=col, color=cores.get(col, COR_CINZA),
                   edgecolor='black', linewidth=0.5)
            base += ct[col].values
        ax.set_ylabel('Nº de obras')
        ax.legend(title='Confiança')
        ax.grid(axis='x', visible=False)
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
        salvar_fig(fig, 'fig_14_spice_label')


# =====================================================================
# EXPORTS
# =====================================================================

def exportar_tags_frequency(stats_tags: Dict) -> None:
    log.info("\n[EXPORT] Exportando frequências de tags...")
    rows = []
    for cat, s in stats_tags.items():
        for tag, count in s['frequencias'].most_common():
            rows.append({'Tipo_Tag': cat.replace('_', ' '), 'Tag': tag, 'Frequencia': count,
                         'Percentual_%': (count / s['n_total']) * 100 if s['n_total'] else 0})
    pd.DataFrame(rows).sort_values('Frequencia', ascending=False).to_csv(
        OUTPUT_DIR / 'tags_frequency.csv', index=False, encoding='utf-8')
    log.info(f"  Salvo: tags_frequency.csv ({len(rows)} tags)")


def exportar_auditoria_titulos(df: pd.DataFrame) -> None:
    if 'Titulo_Limpo' not in df.columns:
        return
    alt = df[df['Titulo_Original_Bruto'] != df['Titulo_Limpo']][['Titulo_Original_Bruto', 'Titulo_Limpo']]
    alt.to_csv(OUTPUT_DIR / 'titulos_limpeza_auditoria.csv', index=False, encoding='utf-8')
    log.info(f"Títulos alterados pela limpeza: {len(alt)}/{len(df)}")
    resto = df[df['Titulo_Limpo'].str.contains(r"\s[–—-]\s|\(|\)|:", regex=True, na=False)]
    if len(resto):
        log.info(f"Títulos ainda com separador/parêntese ({len(resto)}); confira se algum é marketing:")
        for t in resto['Titulo_Limpo'].head(15):
            log.info(f"    {t}")


# =====================================================================
# RELATÓRIO
# =====================================================================

def gerar_relatorio_consolidado(df: pd.DataFrame, stats_tags: Dict, stats_dist: Dict, stats_cat: Dict) -> None:
    log.info("\n[REPORT] Gerando relatório consolidado...")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(OUTPUT_DIR / 'eda_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\nANÁLISE EXPLORATÓRIA DE DADOS (EDA)\n")
        f.write("Dataset: PT-BR Consolidado (livros_ptbr_v3.csv)\n")
        f.write(f"Gerado em: {ts}\n" + "=" * 90 + "\n\n")

        m = META_FILTRO
        f.write("FILTRO TEMPORAL\n" + "-" * 90 + "\n")
        f.write(f"Critério: ano da edição PT-BR >= {m.get('ano_min')}\n")
        f.write(f"Edições no CSV: {m.get('n_inicial')} | removidas por ano < {m.get('ano_min')}: {m.get('n_antes_ano_min')} | "
                f"removidas por ano ausente/inparseável: {m.get('n_sem_ano')} | "
                f"removidas por formato fora de escopo (mangá/HQ/infantil): {m.get('n_formato_excluido', 0)} | "
                f"mantidas: {m.get('n_final')}\n\n")

        f.write("VISÃO GERAL DO DATASET (após filtro)\n" + "-" * 90 + "\n")
        f.write(f"Total de registros: {len(df)}\nTotal de colunas: {len(df.columns)}\n")
        if 'ano' in stats_dist:
            f.write(f"Período temporal (ano da edição PT-BR): {stats_dist['ano']['min']}-{stats_dist['ano']['max']}\n")
        col_ed = 'Editora' if 'Editora' in df.columns else 'Editora_PT'
        if col_ed in df.columns:
            f.write(f"Cobertura de Editora: {df[col_ed].notna().mean() * 100:.1f}%\n")
        col_isbn = 'ISBN_PT' if 'ISBN_PT' in df.columns else 'ISBN'
        if col_isbn in df.columns:
            f.write(f"Cobertura de ISBN: {df[col_isbn].notna().mean() * 100:.1f}%\n")
        f.write("\n")

        for chave, titulo in (('rating', 'RATINGS (por obra)'), ('qtd_avaliacoes', 'QTD DE AVALIAÇÕES (por obra)'),
                              ('paginas', 'PÁGINAS (por edição)')):
            if chave in stats_dist:
                r = stats_dist[chave]
                f.write(titulo + "\n" + "-" * 90 + "\n")
                f.write(f"  Média: {r['mean']:.3f} | Mediana: {r['median']:.3f} | DP: {r['std']:.3f} | "
                        f"Q1-Q3: {r['q1']:.2f}-{r['q3']:.2f} | Mín-Máx: {r['min']:.2f}-{r['max']:.2f} | N: {r['count']}\n")
                if 'ic_boot' in r:
                    f.write(f"  IC bootstrap 95% da média: [{r['ic_boot'][0]:.3f}, {r['ic_boot'][1]:.3f}]\n")
                f.write("\n")

        f.write("ANÁLISE DE TAGS E METADADOS\n" + "-" * 90 + "\n")
        for cat in ['Generos', 'Moods', 'Tags_Gerais', 'Tags_Estruturadas', 'Content_Warnings']:
            if cat not in stats_tags:
                continue
            s = stats_tags[cat]
            f.write(f"\n  {cat.upper()}\n    Tags únicas: {s['n_unicas']}\n    Total de tags: {s['n_total']}\n")
            f.write(f"    Registros com tag: {s['n_registros']}/{s['n_base']} obras ({s['cobertura_%']:.1f}%)\n    Top 10:\n")
            for i, (tag, fr) in enumerate(s['frequencias'].most_common(10), 1):
                pct = fr / s['n_total'] * 100 if s['n_total'] else 0
                f.write(f"      {i:2d}. {tag:<40} {fr:3d} ({pct:5.1f}%)\n")
        f.write("\n" + "=" * 90 + "\n")


def anexar_secao_v3(df: pd.DataFrame, extra: Dict, tab_ngrams: Dict[str, pd.DataFrame]) -> None:
    ob = _por_obra(df)
    pct = lambda a, b: f"{100 * a / b:.1f}%" if b else "n/a"
    with open(OUTPUT_DIR / 'eda_report.txt', 'a', encoding='utf-8') as f:
        f.write("\nSEÇÃO v3: UNIDADES, PONTE E SINAIS DE SPICE\n" + "-" * 90 + "\n")
        f.write(f"Edições (ISBN): {len(df)} | Obras únicas: {len(ob)}\n")
        for col in ('Metodo_Ponte', 'E_Ebook', 'Tipo_Editorial_Editora'):
            if col in df.columns:
                f.write(f"{col}: {df[col].value_counts(dropna=False).to_dict()}\n")
        if 'Revisar_Ponte' in df.columns:
            f.write(f"Revisar_Ponte (entra na auditoria Gold Standard): {int(df['Revisar_Ponte'].sum())}\n")
        for col, rot in (('Estratos_Origem', 'Estratos de origem (quadro amostral proposital)'),
                         ('Origens_Pais', 'Origens (país da autora no seed)'),
                         ('Tipos_Origem', "Tipos de origem (canal do seed; base da sensibilidade sem 'termo:conteudo')")):
            if col in df.columns:
                v = df[col].fillna('(sem)').str.split(r' \| ').explode().value_counts().to_dict()
                f.write(f"{rot}: {v}\n")

        if 'Editora_Chave' in df.columns:
            col_ed = next((c for c in ('Editora_PT', 'Editora') if c in df.columns), None)
            if col_ed:
                f.write(f"\nEditoras: {df[col_ed].dropna().nunique()} grafias brutas -> "
                        f"{df['Editora_Chave'].dropna().nunique()} chaves canônicas\n")
            if 'Tipo_Editorial_Editora' in df.columns:
                desc = df[df['Tipo_Editorial_Editora'] == 'desconhecido']['Editora_Chave'].value_counts().head(15)
                if len(desc):
                    f.write(f"Editoras 'desconhecido' mais frequentes: {desc.to_dict()}\n")

        f.write("\nFlags por obra (proporção com IC de Wilson 95%):\n")
        prev = prevalencias_flags(df)
        for _, r in prev.iterrows():
            f.write(f"  {r['Flag']:<24} {int(r['k']):>4}/{int(r['n'])} = {100 * r['prop']:5.1f}%  "
                    f"[{100 * r['IC95_inf']:.1f}, {100 * r['IC95_sup']:.1f}]\n")
        prev.to_csv(OUTPUT_DIR / 'flags_prevalencia_wilson.csv', index=False, encoding='utf-8')

        if 'Spice_Status' in ob.columns:
            f.write("\nSPICE TRI-ESTADO (por obra). 'sem_sinal' = DESCONHECIDO, não é ausência de spice:\n")
            f.write(f"  Spice_Status: {ob['Spice_Status'].value_counts(dropna=False).to_dict()}\n")
            k = int((ob['Spice_Status'] == 'positivo').sum())
            p, lo, hi = wilson_ic(k, len(ob))
            f.write(f"  Spice positivo: {k}/{len(ob)} = {100 * p:.1f}% [IC Wilson 95%: {100 * lo:.1f}, {100 * hi:.1f}]\n")
            if 'Spice_Nivel' in ob.columns:
                niv = pd.to_numeric(ob['Spice_Nivel'], errors='coerce').value_counts(dropna=False).sort_index()
                niv = {('sem_sinal' if pd.isna(a) else int(a)): int(n) for a, n in niv.items()}
                f.write(f"  Spice_Nivel (0 fechado, 1 sensual, 2 cena aberta, 3 explícito): {niv}\n")
            if 'Flag_Spice_Tag' in ob.columns:
                f.write("  Flag_Spice_Tag (antigo) x Spice_Status (novo):\n    "
                        + pd.crosstab(ob['Flag_Spice_Tag'], ob['Spice_Status']).to_string().replace("\n", "\n    ") + "\n")
            sens = sensibilidade_min_count(ob)
            if sens:
                f.write("  Sensibilidade ao limiar mínimo de votos por tag:\n")
                for kk, vv in sens.items():
                    f.write(f"    {kk}: {vv}\n")
            cob = cobertura_por_popularidade(ob)
            if cob is not None:
                cob.to_csv(OUTPUT_DIR / 'cobertura_por_popularidade.csv', encoding='utf-8')
                f.write("  Cobertura de tags por quartil de popularidade (queda nos quartis baixos = faltante NÃO aleatório):\n    "
                        + cob.to_string().replace("\n", "\n    ") + "\n")
            if 'spearman_users_ntags' in extra:
                rho, pv, n = extra['spearman_users_ntags']
                f.write(f"  Spearman (nº de leitores x nº total de tags): rho={rho:.3f}, p={pv:.4f}, n={n}\n")
            if 'spice_por_pop' in extra:
                f.write("  Spice positivo por quartil de popularidade (Wilson 95%):\n    "
                        + extra['spice_por_pop'].round(3).to_string(index=False).replace("\n", "\n    ") + "\n")
                extra['spice_por_pop'].to_csv(OUTPUT_DIR / 'spice_por_popularidade.csv', index=False, encoding='utf-8')

        if 'Spice_Label' in ob.columns:
            f.write("\nRÓTULO TRIANGULADO DE SPICE (rotulagem.py; supervisão fraca; 'indeterminado' = desconhecido)\n")
            f.write(f"  Spice_Label: {ob['Spice_Label'].value_counts().to_dict()}\n")
            f.write("  Rótulo x confiança:\n    "
                    + pd.crosstab(ob['Spice_Label'], ob['Spice_Label_Confianca']).to_string().replace("\n", "\n    ") + "\n")
            nrev = int((ob['Spice_Label_Confianca'] == 'conflito').sum())
            f.write(f"  Obras em conflito entre fontes (revisão manual): {nrev}\n")

        if {'Flag_Spice_Tag', 'Tipo_Editorial_Editora'} <= set(df.columns):
            f.write("\nAssociação DESCRITIVA (não causal; amostra proposital, não representa o mercado):\n"
                    "Contagem de EDIÇÕES por Flag_Spice_Tag e tipo editorial (atributo da obra replicado por edição)\n")
            tab = pd.crosstab(df['Tipo_Editorial_Editora'].fillna('NA'), df['Flag_Spice_Tag'])
            tab['n'] = tab.sum(axis=1)
            f.write(tab.to_string() + "\n")
        if {'Spice_Status', 'Tipo_Editorial_Editora', 'Hardcover_Book_ID'} <= set(df.columns):
            misto = int((df.groupby('Hardcover_Book_ID')['Tipo_Editorial_Editora'].nunique() > 1).sum())
            f.write(f"\nContagem de OBRAS por Spice_Status e tipo editorial (tipo da 1ª edição; {misto} obra(s) com tipos mistos):\n")
            tab2 = pd.crosstab(ob['Tipo_Editorial_Editora'].fillna('NA'), ob['Spice_Status'])
            tab2['n'] = tab2.sum(axis=1)
            f.write(tab2.to_string() + "\n")

        if tab_ngrams:
            f.write("\nTERMOS MAIS FREQUENTES (frequência de documento, nível obra; lista completa em ngrams_*.csv)\n")
            for nome, t in tab_ngrams.items():
                for tipo in ('unigrama', 'bigrama'):
                    top = t[t['Tipo'] == tipo].head(10)
                    f.write(f"  {nome}/{tipo}: " + ", ".join(f"{a} ({b})" for a, b in zip(top['Termo'], top['Freq_Doc'])) + "\n")
    log.info("  Seção v3 anexada ao eda_report.txt")


# =====================================================================
# MAIN
# =====================================================================

def main():
    global FIG_TITULOS, FORMATOS
    ap = argparse.ArgumentParser(description="EDA BookTok PT-BR (v3.2)")
    ap.add_argument('--ano-min', type=int, default=ANO_MIN_PADRAO, help='ano mínimo da edição PT-BR (padrão 2015)')
    ap.add_argument('--titulos', action='store_true', help='inclui títulos dentro das figuras')
    ap.add_argument('--formatos', nargs='+', default=list(FORMATOS), choices=['png', 'pdf', 'svg'])
    args, _ = ap.parse_known_args()
    FIG_TITULOS, FORMATOS = args.titulos, tuple(args.formatos)
    aplicar_estilo()

    log.info(f"EDA: PT-BR Consolidado | POC I | filtro ano >= {args.ano_min}")
    df = carregar_e_limpar_dados(CAMINHO_ENTRADA, args.ano_min)
    if df.empty:
        log.error("Sem dados após carregamento/filtro. Abortando.")
        sys.exit(1)

    stats_tags = analisar_tags(_por_obra(df))
    stats_dist = analisar_distribuicoes(df)
    stats_cat = analisar_categoricas(df)
    extra = analisar_spice_cobertura(df)

    tab_ngrams = gerar_wordclouds(df)
    gerar_visualizacoes(df, stats_tags, stats_dist, stats_cat, extra)

    exportar_tags_frequency(stats_tags)
    exportar_auditoria_titulos(df)

    gerar_relatorio_consolidado(df, stats_tags, stats_dist, stats_cat)
    anexar_secao_v3(df, extra, tab_ngrams)

    log.info("\n" + "=" * 90 + "\nPIPELINE DE EDA CONCLUÍDO\n" + "=" * 90)
    log.info(f"Outputs em: {OUTPUT_DIR} (figuras: {', '.join(FORMATOS)})")


if __name__ == "__main__":
    main()