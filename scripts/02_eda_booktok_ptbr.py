"""
02_eda_booktok_ptbr.py

Pipeline de Análise Exploratória de Dados (EDA) para Dataset BookTok PT-BR Consolidado
Foco: Engenharia de Features, Padronização, Exploração Multidimensional

Análises Realizadas:
  1. Limpeza e padronização de tags (Gêneros, Moods, Tags Gerais, Content Warnings)
  2. Exploração de distribuições (Rating, Ano, Editora, Fonte de Match)
  3. Análise textual (Nuvem de Palavras para Títulos e Sinopses)
  4. Correlações e associações
  5. Relatório descritivo com estatísticas
  6. (v3.1) Editoras canônicas, spice tri-estado, cobertura de tags por popularidade e
     sensibilidade ao limiar de votos

Uso:
  python scripts/02_eda_booktok_ptbr.py

Outputs:
  - outputs/eda_report.txt (Relatório consolidado)
  - outputs/fig_*.png (Visualizações: distribuições, wordclouds, heatmaps)
  - outputs/tags_frequency.csv (Frequência de todas as tags)
  - outputs/titulos_limpeza_auditoria.csv (Títulos alterados pela limpeza, antes/depois)
  - outputs/editoras_normalizacao_auditoria.csv (grafias de editora fundidas, para conferência humana)
  - outputs/editoras_pares_suspeitos.csv (chaves em que uma é prefixo da outra: mesma editora ou selos distintos?)
  - outputs/cobertura_por_popularidade.csv (tags por quartil de popularidade: checagem de dado faltante não aleatório)
"""

import os
import sys
import warnings
import logging
import ast
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional
import re
import difflib
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from wordcloud import WordCloud
from scipy.stats import spearmanr

# Utilidades compartilhadas com o seed/ponte (mesma limpar_titulo nos dois scripts)
from ptbr_common import limpar_titulo, sem_acento as _sem_acento
from editoras_norm import chave_editora, pares_suspeitos

# Suprimir warnings e configurar matplotlib
warnings.filterwarnings('ignore')
matplotlib.rcParams['figure.figsize'] = (12, 6)
matplotlib.rcParams['font.family'] = 'sans-serif'
sns.set_style("whitegrid")
plt.rcParams['axes.unicode_minus'] = False

# Diretórios
BASE_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = BASE_DIR.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Arquivo de entrada
CAMINHO_ENTRADA = BASE_DIR.parent / "data" / "processed" / "livros_ptbr_v3.csv"

# =====================================================================
# CONFIGURAÇÃO DE LOGGING
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
    """Normaliza string: strip, lower, remove espaços múltiplos."""
    if pd.isna(s) or s is None:
        return ""
    return str(s).strip().lower()


def explodir_tags(tags_input: Optional[object]) -> List[str]:
    """
    Explode tags que podem vir como listas em string (ex: "['tag1', 'tag2']")
    ou separadas por vírgula.
    """
    if pd.isna(tags_input) or not tags_input or tags_input == "":
        return []

    tags_str = str(tags_input).strip()

    # Se estiver no formato de lista Python "['tag1', 'tag2']"
    if tags_str.startswith("[") and tags_str.endswith("]"):
        try:
            lista_eval = ast.literal_eval(tags_str)
            if isinstance(lista_eval, list):
                return [normalizar_string(t) for t in lista_eval if normalizar_string(t)]
        except (ValueError, SyntaxError):
            pass

    # v3: separador " | " (tags podem conter vírgula); v2: vírgula
    tags = tags_str.split("|") if "|" in tags_str else tags_str.split(",")
    return [normalizar_string(t) for t in tags if normalizar_string(t)]


def converter_ano_para_int(ano_str: Optional[object]) -> Optional[int]:
    """Converte string/float de ano para int."""
    if pd.isna(ano_str) or not ano_str:
        return None

    ano_clean = str(ano_str).strip()
    match = re.search(r'\d{4}', ano_clean)
    if match:
        try:
            return int(match.group())
        except ValueError:
            return None
    return None


def normalizar_rating(rating: object) -> Optional[float]:
    """Arredonda rating para 1 casa decimal."""
    if pd.isna(rating) or rating == "":
        return None
    try:
        val = float(rating)
        return round(val, 1)
    except ValueError:
        return None


def _por_obra(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unidade de análise: rating, tags, content warnings e flags são atributos da OBRA
    (Hardcover_Book_ID); capa, editora, ano e páginas são da EDIÇÃO (ISBN). Estatísticas de
    obra usam esta visão deduplicada, senão obras com várias edições pesam mais.
    """
    return df.drop_duplicates('Hardcover_Book_ID') if 'Hardcover_Book_ID' in df.columns else df


def canonizar_editoras(df: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Agrupa grafias da mesma editora por uma chave canônica (editoras_norm.chave_editora), sem alterar
    a coluna bruta. Devolve (df com Editora_Chave e Editora_Canonica, auditoria das chaves que
    fundiram 2+ grafias). Rótulo de exibição = variante mais frequente (empate: a mais curta), sem o
    prefixo 'Editora'.
    """
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
    """
    Taxa de obras com >=1 tag por quartil de popularidade (nº de leitores no Hardcover). Se a cobertura
    cai nos quartis baixos, a ausência de tag acompanha popularidade (dado faltante NÃO aleatório):
    'sem tag' não significa 'sem conteúdo'.
    """
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
    """Limpa texto para nuvem de palavras."""
    if pd.isna(texto) or not texto:
        return ""

    texto = str(texto).lower()
    texto = re.sub(r'http\S+|www\S+', '', texto)
    texto = re.sub(r'["""\'\"""]', '"', texto)
    texto = re.sub(r'[^\w\s\-àáâãäèéêëìíîïòóôõöùúûüçñ]', ' ', texto)

    return texto


# =====================================================================
# STOPWORDS EM PORTUGUÊS
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
    'me', 'te', 'se', 'nos', 'vos', 'meu', 'teu', 'seu', 'nosso', 'vosso',
    'minha', 'tua', 'sua', 'nossa', 'vossa', 'meus', 'teus', 'seus', 'nossos', 'vossos',
    'minhas', 'tuas', 'suas', 'nossas', 'vossas',
    'este', 'esse', 'aquele', 'esta', 'essa', 'aquela',
    'isto', 'isso', 'aquilo', 'estes', 'esses', 'aqueles',
    'é', 'da', 'do', 'das', 'dos', 'vai', 'vão', 'será', 'serão',
    'foi', 'fomos', 'foram', 'era', 'éramos', 'eram',
    'muito', 'pouco', 'todo', 'nenhum', 'algum', 'outro', 'mesmo', 'próprio',
    'mais', 'menos', 'bem', 'mal', 'cá', 'aí', 'lá',
    'já', 'ainda', 'sempre', 'nunca', 'jamais', 'talvez', 'também', 'nem', 'sequer',
    'aqui', 'agora', 'hoje', 'ontem', 'amanhã', 'quando', 'onde', 'porque', 'quem',
}

# Palavras funcionais que a lista acima não cobria: contrações (no, na, ao, pelo...),
# pronomes oblíquos (lhe), advérbios e conjunções frequentes (não, assim, após...),
# numerais e formas de verbos auxiliares.
STOPWORDS_PT_EXTRA = {
    # contrações de preposição + artigo/pronome
    'no', 'na', 'nos', 'nas', 'ao', 'aos', 'à', 'às', 'pelo', 'pela', 'pelos', 'pelas',
    'num', 'numa', 'nuns', 'numas', 'dum', 'duma', 'duns', 'dumas',
    'dele', 'dela', 'deles', 'delas', 'neste', 'nesta', 'nestes', 'nestas',
    'nesse', 'nessa', 'nesses', 'nessas', 'naquele', 'naquela', 'naqueles', 'naquelas',
    'nisto', 'nisso', 'naquilo', 'deste', 'desta', 'destes', 'destas',
    'desse', 'dessa', 'desses', 'dessas', 'daquele', 'daquela', 'daqueles', 'daquelas',
    'disto', 'disso', 'daquilo', 'àquele', 'àquela', 'àqueles', 'àquelas',
    # pronomes
    'lhe', 'lhes', 'mim', 'ti', 'si', 'consigo', 'comigo', 'contigo', 'conosco', 'convosco',
    'você', 'vocês', 'outra', 'outros', 'outras', 'tal', 'tais', 'qual', 'quais', 'quê',
    'cujo', 'cuja', 'cujos', 'cujas', 'quanto', 'quanta', 'quantos', 'quantas',
    'alguém', 'ninguém', 'tudo', 'nada', 'algo', 'cada', 'vários', 'várias', 'alguns', 'algumas',
    'nenhuma', 'nenhuns', 'nenhumas', 'toda', 'todos', 'todas', 'certo', 'certa',
    # advérbios, conjunções e preposições
    'não', 'sim', 'assim', 'apenas', 'só', 'somente', 'depois', 'antes', 'após', 'então',
    'enquanto', 'embora', 'apesar', 'entanto', 'tanto', 'tão', 'quase', 'tampouco',
    'ali', 'além', 'através', 'sob', 'contra', 'ante', 'após', 'desde', 'durante',
    'logo', 'inclusive', 'inclusive', 'ainda', 'cada', 'sendo', 'pois', 'caso', 'dado',
    'pra', 'pro', 'pras', 'pros', 'né', 'tá',
    # numerais
    'dois', 'duas', 'três',
    # verbos auxiliares / de alta frequência sem valor temático
    'sido', 'sendo', 'fui', 'fora', 'seja', 'sejam', 'seria', 'serem', 'sendo',
    'estava', 'estavam', 'esteja', 'estive', 'estiver',
    'tinha', 'tinham', 'tenha', 'tenham', 'tido', 'tendo', 'teve', 'tiver',
    'há', 'havia', 'houve', 'haja',
    'pode', 'podem', 'podia', 'podiam', 'possa', 'possam', 'pôde',
    'deve', 'devem', 'devia', 'vem', 'vêm', 'veio', 'vindo', 'vão', 'vamos', 'ia', 'iam',
    'quer', 'querem', 'queria', 'queriam', 'quis',
}
STOPWORDS_PT = STOPWORDS_PT | STOPWORDS_PT_EXTRA
# Inclui as versões sem acento, pois alguns títulos/sinopses chegam sem acentuação.
STOPWORDS_PT = STOPWORDS_PT | {_sem_acento(w) for w in STOPWORDS_PT}

# Várias edições brasileiras mantêm o título em inglês ("The Risk", "Part of Your World"):
# sem isto, "the" e "of" dominam a nuvem de títulos.
STOPWORDS_EN = {
    'the', 'of', 'and', 'to', 'in', 'on', 'at', 'for', 'with', 'a', 'an',
    'is', 'it', 'my', 'your', 'you', 'we', 'our', 'his', 'her', 'its',
}


# =====================================================================
# CARREGAMENTO E LIMPEZA DE DADOS
# =====================================================================

def carregar_e_limpar_dados(caminho: Path) -> pd.DataFrame:
    """Carrega CSV adaptado para o schema de livros_ptbr_v2.csv."""
    log.info(f"Carregando dados de {caminho}...")

    if not caminho.exists():
        log.error(f"Arquivo não encontrado: {caminho}")
        return pd.DataFrame()

    df = pd.read_csv(caminho)
    log.info(f"Carregados {len(df)} registros com {len(df.columns)} colunas")

    # Editoras: chave canônica (a coluna bruta é preservada; auditoria exportada para conferência humana)
    col_ed_raw = next((c for c in ('Editora_PT', 'Editora') if c in df.columns), None)
    if col_ed_raw:
        n_bruto = df[col_ed_raw].dropna().nunique()
        df, aud_ed = canonizar_editoras(df, col_ed_raw)
        aud_ed.to_csv(OUTPUT_DIR / 'editoras_normalizacao_auditoria.csv', encoding='utf-8')
        n_canon = df['Editora_Chave'].dropna().nunique()
        log.info(f"Editoras: {n_bruto} grafias brutas -> {n_canon} canônicas "
                 f"({len(aud_ed)} grupos fundiram 2+ grafias; editoras_normalizacao_auditoria.csv)")
        sus = pares_suspeitos(df['Editora_Chave'].dropna().unique())
        pd.DataFrame(sus, columns=['Chave_Curta', 'Chave_Longa']).to_csv(
            OUTPUT_DIR / 'editoras_pares_suspeitos.csv', index=False, encoding='utf-8')
        if sus:
            log.info(f"Editoras: {len(sus)} pares suspeitos (mesma editora ou selos distintos?) "
                     f"em editoras_pares_suspeitos.csv; decida e registre em ALIASES_EDITORA")

    # Normalização de Rating
    if 'Rating' in df.columns:
        df['Rating_Normalizado'] = df['Rating'].apply(normalizar_rating)
    elif 'Avaliacao_Hardcover' in df.columns:
        df['Rating_Normalizado'] = df['Avaliacao_Hardcover'].apply(normalizar_rating)

    # Conversão de Quantidade de Avaliações
    if 'Qtd_Avaliacoes' in df.columns:
        df['Qtd_Avaliacoes_Clean'] = pd.to_numeric(df['Qtd_Avaliacoes'], errors='coerce')
    elif 'Qtd_Avaliacoes_Hardcover' in df.columns:
        df['Qtd_Avaliacoes_Clean'] = pd.to_numeric(df['Qtd_Avaliacoes_Hardcover'], errors='coerce')

    # Conversão de Número de Páginas
    if 'Paginas' in df.columns:
        df['Num_Paginas_Clean'] = pd.to_numeric(df['Paginas'], errors='coerce')
    elif 'Num_Paginas_PT' in df.columns:
        df['Num_Paginas_Clean'] = pd.to_numeric(df['Num_Paginas_PT'], errors='coerce')

    # Ano de publicação da edição PT-BR (não o ano da obra original)
    col_ano = next((c for c in ('Ano_Publicacao_PT', 'Ano_Publicacao') if c in df.columns), None)
    if col_ano:
        df['Ano'] = df[col_ano].apply(converter_ano_para_int)
        log.info(f"Ano PT disponível em {int(df['Ano'].notna().sum())}/{len(df)} registros")
    else:
        log.warning("Coluna de ano PT (Ano_Publicacao_PT) não encontrada; análise temporal ignorada")

    # Padronização de Colunas de Tags
    mapeamento_tags = {
        'Generos': 'Generos',
        'Generos_Hardcover': 'Generos',
        'Moods': 'Moods',
        'Moods_Hardcover': 'Moods',
        'Tags_Gerais': 'Tags_Gerais',
        'Tags_Gerais_Hardcover': 'Tags_Gerais',
        'Content_Warnings': 'Content_Warnings'
    }

    for col_orig, col_destino in mapeamento_tags.items():
        if col_orig in df.columns:
            df[f'{col_destino}_list'] = df[col_orig].apply(explodir_tags)

    # Título limpo (sem marketing/edição/série) e texto para wordcloud
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
# ANÁLISES ESPECÍFICAS
# =====================================================================

def analisar_tags(df: pd.DataFrame) -> Dict[str, Dict]:
    """Análise consolidada de tags: frequência e cobertura."""
    log.info("\n[ANÁLISE 1] Processando Tags...")

    resultados = {}
    categorias_tags = ['Generos', 'Moods', 'Tags_Gerais', 'Content_Warnings']

    for cat in categorias_tags:
        col_list = f'{cat}_list'

        if col_list not in df.columns:
            continue

        todas_tags = []
        for tags_list in df[col_list]:
            if isinstance(tags_list, list):
                todas_tags.extend(tags_list)

        freq = Counter(todas_tags)
        n_unicas = len(freq)
        n_total = len(todas_tags)
        n_registros_com_tag = df[col_list].apply(lambda x: isinstance(x, list) and len(x) > 0).sum()

        resultados[cat] = {
            'frequencias': freq,
            'n_unicas': n_unicas,
            'n_total': n_total,
            'n_registros': n_registros_com_tag,
            'cobertura_%': (n_registros_com_tag / len(df)) * 100 if len(df) > 0 else 0,
            'n_base': len(df),
        }

        log.info(f"  {cat}: {n_unicas} tags únicas, "
                 f"{n_registros_com_tag}/{len(df)} registros ({resultados[cat]['cobertura_%']:.1f}%)")

    return resultados


def analisar_distribuicoes(df: pd.DataFrame) -> Dict:
    """Análise de distribuições contínuas: rating, ano, páginas, etc."""
    log.info("\n[ANÁLISE 2] Distribuições Contínuas...")

    resultados = {}

    # Rating
    if 'Rating_Normalizado' in df.columns:
        rating_data = _por_obra(df)['Rating_Normalizado'].dropna()
        if len(rating_data) > 0:
            resultados['rating'] = {
                'mean': rating_data.mean(),
                'median': rating_data.median(),
                'std': rating_data.std(),
                'min': rating_data.min(),
                'max': rating_data.max(),
                'count': len(rating_data),
            }
            log.info(f"  Rating: média={resultados['rating']['mean']:.2f}, "
                     f"dp={resultados['rating']['std']:.2f}, n={resultados['rating']['count']}")

    # Qtd Avaliações
    if 'Qtd_Avaliacoes_Clean' in df.columns:
        qtd_data = _por_obra(df)['Qtd_Avaliacoes_Clean'].dropna()
        if len(qtd_data) > 0:
            resultados['qtd_avaliacoes'] = {
                'mean': qtd_data.mean(),
                'median': qtd_data.median(),
                'std': qtd_data.std(),
                'min': qtd_data.min(),
                'max': qtd_data.max(),
                'count': len(qtd_data),
            }
            log.info(f"  Qtd Avaliações: média={resultados['qtd_avaliacoes']['mean']:.0f}, "
                     f"dp={resultados['qtd_avaliacoes']['std']:.0f}")

    # Páginas
    if 'Num_Paginas_Clean' in df.columns:
        pag_data = df[df['Num_Paginas_Clean'] > 0]['Num_Paginas_Clean'].dropna()
        if len(pag_data) > 0:
            resultados['paginas'] = {
                'mean': pag_data.mean(),
                'median': pag_data.median(),
                'std': pag_data.std(),
                'min': pag_data.min(),
                'max': pag_data.max(),
                'count': len(pag_data),
            }
            log.info(f"  Páginas: média={resultados['paginas']['mean']:.0f}, "
                     f"dp={resultados['paginas']['std']:.0f}")

    # Ano
    if 'Ano' in df.columns:
        ano_data = df['Ano'].dropna()
        if len(ano_data) > 0:
            resultados['ano'] = {
                'mean': ano_data.mean(),
                'median': ano_data.median(),
                'min': int(ano_data.min()),
                'max': int(ano_data.max()),
                'count': len(ano_data),
            }
            log.info(f"  Ano: média={resultados['ano']['mean']:.0f}, "
                     f"range=[{resultados['ano']['min']}, {resultados['ano']['max']}]")

    return resultados


def analisar_categoricas(df: pd.DataFrame) -> Dict:
    """Análise de distribuições categóricas (Editora, Fonte de Match)."""
    log.info("\n[ANÁLISE 3] Distribuições Categóricas...")

    resultados = {}

    # Editora CANÔNICA (grafias agrupadas); cai para a coluna bruta se a canonização não rodou
    col_editora = next((c for c in ('Editora_Canonica', 'Editora', 'Editora_PT') if c in df.columns), None)
    if col_editora:
        editoras = df[col_editora].dropna()
        resultados['editora'] = editoras.value_counts().to_dict()
        log.info(f"  Editoras ({col_editora}): {len(resultados['editora'])} únicas, "
                 f"cobertura {(len(editoras)/len(df))*100:.1f}%")
        for ed, cnt in list(resultados['editora'].items())[:5]:
            log.info(f"      - {ed}: {cnt}")

    col_fonte = next((c for c in ('Metodo_Ponte', 'Fonte_Match') if c in df.columns), None)
    if col_fonte:
        fontes = df[col_fonte].value_counts().to_dict()
        resultados['fonte_match'] = fontes
        log.info(f"  Fonte de Match: {len(fontes)} tipos")
        for fnt, cnt in fontes.items():
            log.info(f"      - {fnt}: {cnt}")

    return resultados


def gerar_wordclouds(df: pd.DataFrame, output_dir: Path) -> None:
    """Gera nuvens de palavras para títulos e sinopses."""
    log.info("\n[ANÁLISE 4] Gerando Word Clouds...")

    # Wordcloud de Títulos
    if 'Titulo_clean' in df.columns:
        titulo_text = " ".join(df['Titulo_clean'].dropna().tolist())
        if titulo_text.strip():
            log.info("  Gerando wordcloud de TÍTULOS...")
            fig, ax = plt.subplots(figsize=(14, 8))
            wc_titulo = WordCloud(
                width=1400,
                height=800,
                background_color='white',
                colormap='RdPu',
                stopwords=STOPWORDS_PT | STOPWORDS_EN,
                min_word_length=2,
                relative_scaling=0.5,
                min_font_size=10,
                prefer_horizontal=0.7,
            ).generate(titulo_text)
            ax.imshow(wc_titulo, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Nuvem de Palavras: Títulos dos Livros PT-BR', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(output_dir / 'fig_wordcloud_titulos.png', dpi=300, bbox_inches='tight')
            plt.close()
            log.info("    Salvo em fig_wordcloud_titulos.png")

    # Wordcloud de Sinopses
    if 'Sinopse_clean' in df.columns:
        sinopse_text = " ".join(df['Sinopse_clean'].dropna().tolist())
        if sinopse_text.strip():
            log.info("  Gerando wordcloud de SINOPSES...")
            fig, ax = plt.subplots(figsize=(14, 8))
            wc_sinopse = WordCloud(
                width=1400,
                height=800,
                background_color='white',
                colormap='viridis',
                stopwords=STOPWORDS_PT | STOPWORDS_EN,
                min_word_length=2,
                relative_scaling=0.5,
                min_font_size=8,
                max_words=100,
                prefer_horizontal=0.7,
            ).generate(sinopse_text)
            ax.imshow(wc_sinopse, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Nuvem de Palavras: Sinopses dos Livros PT-BR', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(output_dir / 'fig_wordcloud_sinopses.png', dpi=300, bbox_inches='tight')
            plt.close()
            log.info("    Salvo em fig_wordcloud_sinopses.png")


def gerar_visualizacoes(df: pd.DataFrame, output_dir: Path, stats_tags: Dict, stats_dist: Dict, stats_cat: Dict) -> None:
    """Gera visualizações gráficas completas."""
    log.info("\n[ANÁLISE 5] Gerando Visualizações...")

    # FIG 1: Distribuição de Ratings
    if 'Rating_Normalizado' in df.columns:
        rating_data = _por_obra(df)['Rating_Normalizado'].dropna()
        if len(rating_data) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            axes[0].hist(rating_data, bins=10, color='steelblue', edgecolor='black', alpha=0.7)
            axes[0].axvline(rating_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Média: {rating_data.mean():.2f}')
            axes[0].axvline(rating_data.median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {rating_data.median():.2f}')
            axes[0].set_xlabel('Rating', fontsize=11)
            axes[0].set_ylabel('Frequência', fontsize=11)
            axes[0].set_title('Distribuição de Ratings', fontsize=12, fontweight='bold')
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            axes[1].boxplot(rating_data, vert=True)
            axes[1].set_ylabel('Rating', fontsize=11)
            axes[1].set_title('Boxplot: Ratings', fontsize=12, fontweight='bold')
            axes[1].grid(alpha=0.3, axis='y')

            plt.tight_layout()
            plt.savefig(output_dir / 'fig_01_ratings_dist.png', dpi=300, bbox_inches='tight')
            plt.close()
            log.info("  Salvo: fig_01_ratings_dist.png")

    # FIG 2: Distribuição de Anos
    if 'Ano' in df.columns:
        ano_data = df['Ano'].value_counts().sort_index()
        if len(ano_data) > 0:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(ano_data.index.astype(str), ano_data.values, color='coral', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Ano de publicação da edição PT-BR', fontsize=11)
            ax.set_ylabel('Quantidade de Livros', fontsize=11)
            ax.set_title('Distribuição temporal: ano de publicação da edição PT-BR', fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3, axis='y')

            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'fig_02_anos_dist.png', dpi=300, bbox_inches='tight')
            plt.close()
            log.info("  Salvo: fig_02_anos_dist.png")

    # FIG 3: Top 10 Editoras
    if 'editora' in stats_cat and len(stats_cat['editora']) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        editoras_top = dict(sorted(stats_cat['editora'].items(), key=lambda x: x[1], reverse=True)[:10])
        ax.barh(list(editoras_top.keys()), list(editoras_top.values()), color='lightseagreen', edgecolor='black')
        ax.set_xlabel('Quantidade de Edições', fontsize=11)
        ax.set_title('Top 10 Editoras no Dataset (grafias agrupadas)', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(output_dir / 'fig_03_editoras_top10.png', dpi=300, bbox_inches='tight')
        plt.close()
        log.info("  Salvo: fig_03_editoras_top10.png")

    # FIG 4: Top Gêneros, Moods, Tags
    col_tags_to_viz = ['Generos', 'Moods', 'Tags_Gerais']
    n_subplots = len([c for c in col_tags_to_viz if c in stats_tags])

    if n_subplots > 0:
        fig, axes = plt.subplots(n_subplots, 1, figsize=(12, 4 * n_subplots))
        if n_subplots == 1:
            axes = [axes]

        idx = 0
        for col in col_tags_to_viz:
            if col not in stats_tags:
                continue

            freq = stats_tags[col]['frequencias']
            top_n = 12
            top_tags = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n])

            axes[idx].barh(list(top_tags.keys()), list(top_tags.values()),
                          color=plt.cm.Set3(idx), edgecolor='black', alpha=0.8)
            axes[idx].set_xlabel('Frequência', fontsize=10)
            axes[idx].set_title(f'Top {top_n} {col}', fontsize=11, fontweight='bold')
            axes[idx].grid(alpha=0.3, axis='x')
            idx += 1

        plt.tight_layout()
        plt.savefig(output_dir / 'fig_04_tags_top.png', dpi=300, bbox_inches='tight')
        plt.close()
        log.info("  Salvo: fig_04_tags_top.png")

    # FIG 5: Scatter - Rating vs Qtd Avaliações
    if 'Rating_Normalizado' in df.columns and 'Qtd_Avaliacoes_Clean' in df.columns:
        df_plot = _por_obra(df).dropna(subset=['Rating_Normalizado', 'Qtd_Avaliacoes_Clean'])
        if len(df_plot) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            scatter = ax.scatter(df_plot['Qtd_Avaliacoes_Clean'],
                                df_plot['Rating_Normalizado'],
                                s=100, alpha=0.6, c=df_plot['Qtd_Avaliacoes_Clean'],
                                cmap='viridis', edgecolors='black', linewidth=1)

            if len(df_plot) > 2:
                corr, p_value = spearmanr(df_plot['Qtd_Avaliacoes_Clean'], df_plot['Rating_Normalizado'])
                ax.text(0.05, 0.95, f'Correlação Spearman: {corr:.3f} (p={p_value:.3f})',
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            ax.set_xlabel('Quantidade de Avaliações', fontsize=11)
            ax.set_ylabel('Rating', fontsize=11)
            ax.set_title('Relação: Rating vs Quantidade de Avaliações', fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Qtd Avaliações')

            plt.tight_layout()
            plt.savefig(output_dir / 'fig_05_rating_vs_qtd_scatter.png', dpi=300, bbox_inches='tight')
            plt.close()
            log.info("  Salvo: fig_05_rating_vs_qtd_scatter.png")

    # FIG 6: Content Warnings - Top 15
    if 'Content_Warnings' in stats_tags:
        freq_cw = stats_tags['Content_Warnings']['frequencias']
        if len(freq_cw) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            top_cw = dict(sorted(freq_cw.items(), key=lambda x: x[1], reverse=True)[:15])
            colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_cw)))
            ax.barh(list(top_cw.keys()), list(top_cw.values()),
                   color=colors, edgecolor='darkred', linewidth=1.5)
            ax.set_xlabel('Frequência', fontsize=11)
            ax.set_title('Top 15 Content Warnings (Temas Sensíveis)', fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3, axis='x')

            plt.tight_layout()
            plt.savefig(output_dir / 'fig_06_content_warnings.png', dpi=300, bbox_inches='tight')
            plt.close()
            log.info("  Salvo: fig_06_content_warnings.png")

    # FIG 7: Fonte de Match Distribution
    if 'fonte_match' in stats_cat and len(stats_cat['fonte_match']) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        fontes = stats_cat['fonte_match']
        colors_pie = plt.cm.Spectral(np.linspace(0, 1, len(fontes)))

        ax.pie(fontes.values(), labels=fontes.keys(), autopct='%1.1f%%',
               colors=colors_pie, startangle=90, textprops={'fontsize': 10})
        ax.set_title('Distribuição: Fonte de Match (Camadas do Pipeline)', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_dir / 'fig_07_fonte_match_pie.png', dpi=300, bbox_inches='tight')
        plt.close()
        log.info("  Salvo: fig_07_fonte_match_pie.png")


def exportar_tags_frequency(df: pd.DataFrame, output_dir: Path, stats_tags: Dict) -> None:
    """Exporta tabela de frequências de tags para CSV."""
    log.info("\n[EXPORT] Exportando frequências de tags...")

    registros_export = []
    for col_tags, stats in stats_tags.items():
        freq = stats['frequencias']
        for tag, count in freq.most_common():
            registros_export.append({
                'Tipo_Tag': col_tags.replace('_', ' '),
                'Tag': tag,
                'Frequencia': count,
                'Percentual_%': (count / stats['n_total']) * 100 if stats['n_total'] > 0 else 0,
            })

    df_export = pd.DataFrame(registros_export).sort_values('Frequencia', ascending=False)
    output_file = output_dir / 'tags_frequency.csv'
    df_export.to_csv(output_file, index=False, encoding='utf-8')
    log.info(f"  Salvo: tags_frequency.csv ({len(registros_export)} tags)")


def exportar_auditoria_titulos(df: pd.DataFrame, output_dir: Path) -> None:
    """Exporta antes/depois da limpeza de títulos para conferência manual."""
    if 'Titulo_Limpo' not in df.columns:
        return
    alterados = df[df['Titulo_Original_Bruto'] != df['Titulo_Limpo']][['Titulo_Original_Bruto', 'Titulo_Limpo']]
    alterados.to_csv(output_dir / 'titulos_limpeza_auditoria.csv', index=False, encoding='utf-8')
    log.info(f"Títulos alterados pela limpeza: {len(alterados)}/{len(df)} (titulos_limpeza_auditoria.csv)")

    resto = df[df['Titulo_Limpo'].str.contains(r"\s[–—-]\s|\(|\)|:", regex=True, na=False)]
    if len(resto):
        log.info(f"Títulos ainda com separador ou parêntese ({len(resto)}); confira se algum é marketing:")
        for t in resto['Titulo_Limpo'].head(15):
            log.info(f"    {t}")


def gerar_relatorio_consolidado(
    df: pd.DataFrame,
    stats_tags: Dict,
    stats_dist: Dict,
    stats_cat: Dict,
    output_dir: Path
) -> None:
    """Gera relatório consolidado em TXT."""
    log.info("\n[REPORT] Gerando relatório consolidado...")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(output_dir / 'eda_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("ANÁLISE EXPLORATÓRIA DE DADOS (EDA)\n")
        f.write("Dataset: BookTok PT-BR Consolidado (livros_ptbr_v3.csv)\n")
        f.write(f"Gerado em: {timestamp}\n")
        f.write("=" * 90 + "\n\n")

        # ======== OVERVIEW ========
        f.write("VISÃO GERAL DO DATASET\n")
        f.write("-" * 90 + "\n")
        f.write(f"Total de registros: {len(df)}\n")
        f.write(f"Total de colunas: {len(df.columns)}\n")
        if 'ano' in stats_dist:
            f.write(f"Período temporal (ano da edição PT-BR): {stats_dist['ano']['min']}-{stats_dist['ano']['max']}\n")

        col_ed = 'Editora' if 'Editora' in df.columns else 'Editora_PT'
        if col_ed in df.columns:
            f.write(f"Cobertura de Editora: {(df[col_ed].notna().sum() / len(df)) * 100:.1f}%\n")

        col_isbn = 'ISBN_PT' if 'ISBN_PT' in df.columns else 'ISBN'
        if col_isbn in df.columns:
            f.write(f"Cobertura de ISBN: {(df[col_isbn].notna().sum() / len(df)) * 100:.1f}%\n")
        f.write("\n")

        # ======== RATINGS ========
        if 'rating' in stats_dist:
            f.write("RATINGS\n")
            f.write("-" * 90 + "\n")
            r = stats_dist['rating']
            f.write(f"  Média:               {r['mean']:.3f}\n")
            f.write(f"  Mediana:             {r['median']:.3f}\n")
            f.write(f"  Desvio Padrão:      {r['std']:.3f}\n")
            f.write(f"  Mín - Máx:          {r['min']:.3f} - {r['max']:.3f}\n")
            f.write(f"  N registros:        {r['count']}\n")
            f.write("\n")

        # ======== TAGS ========
        f.write("ANÁLISE DE TAGS E METADADOS\n")
        f.write("-" * 90 + "\n")

        for cat in ['Generos', 'Moods', 'Tags_Gerais', 'Content_Warnings']:
            if cat not in stats_tags:
                continue

            s = stats_tags[cat]
            f.write(f"\n  {cat.upper()}\n")
            f.write(f"    Tags únicas:        {s['n_unicas']}\n")
            f.write(f"    Total de tags:      {s['n_total']}\n")
            f.write(f"    Registros com tag:  {s['n_registros']}/{s['n_base']} obras ({s['cobertura_%']:.1f}%)\n")
            f.write(f"    Top 10 mais comuns:\n")

            top_10 = s['frequencias'].most_common(10)
            for i, (tag, freq) in enumerate(top_10, 1):
                pct = (freq / s['n_total']) * 100 if s['n_total'] > 0 else 0
                f.write(f"      {i:2d}. {tag:<40} {freq:3d} ({pct:5.1f}%)\n")

        f.write("\n" + "=" * 90 + "\n")

    log.info(f"  Relatório salvo: eda_report.txt")


# =====================================================================
# MAIN
# =====================================================================

def anexar_secao_v3(df: pd.DataFrame, output_dir: Path) -> None:
    """Seção do dataset v3: unidades, qualidade da ponte, sinais de spice e associações DESCRITIVAS."""
    ob = _por_obra(df)
    pct = lambda a, b: f"{100 * a / b:.1f}%" if b else "n/a"
    with open(output_dir / 'eda_report.txt', 'a', encoding='utf-8') as f:
        f.write("\nSEÇÃO v3: UNIDADES, PONTE E SINAIS DE SPICE\n" + "-" * 90 + "\n")
        f.write(f"Edições (ISBN): {len(df)} | Obras únicas: {len(ob)}\n")
        for col in ('Metodo_Ponte', 'E_Ebook', 'Tipo_Editorial_Editora'):
            if col in df.columns:
                f.write(f"{col}: {df[col].value_counts(dropna=False).to_dict()}\n")
        if 'Revisar_Ponte' in df.columns:
            f.write(f"Revisar_Ponte (entra na auditoria Gold Standard): {int(df['Revisar_Ponte'].sum())}\n")
        if 'Estratos_Origem' in df.columns:
            est = df['Estratos_Origem'].fillna('(sem)').str.split(r' \| ').explode().value_counts().to_dict()
            f.write(f"Estratos de origem (quadro amostral proposital): {est}\n")
            if set(est) <= {'(sem)'}:
                f.write("  ATENÇÃO: nenhum estrato propagado; rode a ponte (backfill_proveniencia) antes de interpretar.\n")
        if 'Origens_Pais' in df.columns:
            ori = df['Origens_Pais'].fillna('(sem)').str.split(r' \| ').explode().value_counts().to_dict()
            f.write(f"Origens (país da autora no seed): {ori}\n")
        if 'Tipos_Origem' in df.columns:
            tip = df['Tipos_Origem'].fillna('(sem)').str.split(r' \| ').explode().value_counts().to_dict()
            f.write(f"Tipos de origem (canal do seed; base da análise de sensibilidade sem 'termo:conteudo'): {tip}\n")

        if 'Editora_Chave' in df.columns:
            col_ed = next((c for c in ('Editora_PT', 'Editora') if c in df.columns), None)
            if col_ed:
                f.write(f"\nEditoras: {df[col_ed].dropna().nunique()} grafias brutas -> "
                        f"{df['Editora_Chave'].dropna().nunique()} chaves canônicas "
                        f"(auditoria: editoras_normalizacao_auditoria.csv; pares_suspeitos: editoras_pares_suspeitos.csv)\n")
            if 'Tipo_Editorial_Editora' in df.columns:
                desc = df[df['Tipo_Editorial_Editora'] == 'desconhecido']['Editora_Chave'].value_counts().head(15)
                if len(desc):
                    f.write("Editoras 'desconhecido' mais frequentes (candidatas a entrar em editoras.txt): "
                            f"{desc.to_dict()}\n")

        f.write("\nFlags por obra:\n")
        for c in ('Flag_Romance', 'Flag_New_Adult', 'Flag_Spice_Tag', 'Flag_CW_Sexual_Content', 'Flag_Closed_Door_Tag', 'Flag_Formato_Excluir'):
            if c in ob.columns:
                f.write(f"  {c:<24} {int(ob[c].sum())} ({pct(ob[c].sum(), len(ob))})\n")

        # ---- spice tri-estado (v3.1)
        if 'Spice_Status' in ob.columns:
            f.write("\nSPICE TRI-ESTADO (por obra). 'sem_sinal' = DESCONHECIDO, não é ausência de spice:\n")
            f.write(f"  Spice_Status: {ob['Spice_Status'].value_counts(dropna=False).to_dict()}\n")
            if 'Spice_Nivel' in ob.columns:
                niveis = pd.to_numeric(ob['Spice_Nivel'], errors='coerce').value_counts(dropna=False).sort_index()
                niveis = {('sem_sinal' if pd.isna(k) else int(k)): int(n) for k, n in niveis.items()}
                f.write(f"  Spice_Nivel (0 fechado, 1 sensual, 2 cena aberta, 3 explícito): {niveis}\n")
            if 'Flag_Spice_Tag' in ob.columns:
                f.write("  Flag_Spice_Tag (antigo, limiar de votos) x Spice_Status (novo):\n")
                f.write("    " + pd.crosstab(ob['Flag_Spice_Tag'], ob['Spice_Status']).to_string().replace("\n", "\n    ") + "\n")
            sens = sensibilidade_min_count(ob)
            if sens:
                f.write("  Sensibilidade ao limiar mínimo de votos por tag:\n")
                for k, vv in sens.items():
                    f.write(f"    {k}: {vv}\n")
            cob = cobertura_por_popularidade(ob)
            if cob is not None:
                cob.to_csv(output_dir / 'cobertura_por_popularidade.csv', encoding='utf-8')
                f.write("  Cobertura de tags por quartil de popularidade (se cai nos quartis baixos, a ausência de "
                        "tag é dado faltante NÃO aleatório):\n")
                f.write("    " + cob.to_string().replace("\n", "\n    ") + "\n")

        if {'Flag_Spice_Tag', 'Tipo_Editorial_Editora'} <= set(df.columns):
            f.write("\nAssociação DESCRITIVA (não causal; amostra proposital, não representa o mercado):\n"
                    "Contagem de EDIÇÕES por Flag_Spice_Tag e tipo editorial (atributo da obra replicado por edição)\n")
            tab = pd.crosstab(df['Tipo_Editorial_Editora'].fillna('NA'), df['Flag_Spice_Tag'])
            tab['n'] = tab.sum(axis=1)
            f.write(tab.to_string() + "\n")
        if {'Spice_Status', 'Tipo_Editorial_Editora', 'Hardcover_Book_ID'} <= set(df.columns):
            misto = int((df.groupby('Hardcover_Book_ID')['Tipo_Editorial_Editora'].nunique() > 1).sum())
            f.write("\nContagem de OBRAS por Spice_Status e tipo editorial (tipo da 1ª edição da obra; "
                    f"{misto} obra(s) têm edições de tipos editoriais mistos):\n")
            tab2 = pd.crosstab(ob['Tipo_Editorial_Editora'].fillna('NA'), ob['Spice_Status'])
            tab2['n'] = tab2.sum(axis=1)
            f.write(tab2.to_string() + "\n")
    log.info("  Seção v3 anexada ao eda_report.txt")


def main():
    """Pipeline completo de EDA."""

    log.info("EDA: BookTok PT-BR Consolidado | POC I - Engenharia de Dados")

    # Carregamento
    df = carregar_e_limpar_dados(CAMINHO_ENTRADA)
    if df.empty:
        log.error("Falha ao carregar dados. Abortando.")
        sys.exit(1)

    # Análises
    stats_tags = analisar_tags(_por_obra(df))
    stats_dist = analisar_distribuicoes(df)
    stats_cat = analisar_categoricas(df)

    # Visualizações
    gerar_wordclouds(df, OUTPUT_DIR)
    gerar_visualizacoes(df, OUTPUT_DIR, stats_tags, stats_dist, stats_cat)

    # Exports
    exportar_tags_frequency(df, OUTPUT_DIR, stats_tags)
    exportar_auditoria_titulos(df, OUTPUT_DIR)

    # Relatório
    gerar_relatorio_consolidado(df, stats_tags, stats_dist, stats_cat, OUTPUT_DIR)
    anexar_secao_v3(df, OUTPUT_DIR)

    # =====================================================================
    # RESUMO FINAL
    # =====================================================================
    log.info("\n" + "="*90)
    log.info("PIPELINE DE EDA CONCLUÍDO COM SUCESSO")
    log.info("="*90)
    log.info(f"\nOutputs gerados em: {OUTPUT_DIR}")
    log.info("\nArquivos:")
    log.info("  eda_report.txt                      relatório consolidado")
    log.info("  fig_01_ratings_dist.png             distribuição de ratings")
    log.info("  fig_02_anos_dist.png                distribuição temporal (ano da edição PT-BR)")
    log.info("  fig_03_editoras_top10.png           top 10 editoras (grafias agrupadas)")
    log.info("  fig_04_tags_top.png                 top tags por categoria")
    log.info("  fig_05_rating_vs_qtd_scatter.png    rating vs quantidade de avaliações")
    log.info("  fig_06_content_warnings.png         content warnings")
    log.info("  fig_07_fonte_match_pie.png          fontes de match (se existir a coluna)")
    log.info("  fig_wordcloud_titulos.png           nuvem de palavras: títulos limpos")
    log.info("  fig_wordcloud_sinopses.png          nuvem de palavras: sinopses")
    log.info("  tags_frequency.csv                  frequência consolidada de tags")
    log.info("  titulos_limpeza_auditoria.csv       títulos alterados pela limpeza")
    log.info("  editoras_normalizacao_auditoria.csv grafias de editora fundidas (conferir)")
    log.info("  editoras_pares_suspeitos.csv        chaves prefixo uma da outra (decidir)")
    log.info("  cobertura_por_popularidade.csv      cobertura de tags por quartil de popularidade")
    log.info("\n" + "="*90 + "\n")


if __name__ == "__main__":
    main()