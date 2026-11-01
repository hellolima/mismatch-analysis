"""Camada canonica de carregamento, filtragem de escopo e deduplicacao por Obra (FRBR).

Unifica a logica antes triplicada entre `rotulagem.py`, `diagnostico_rotulo.py` e `02_eda_booktok_ptbr.py`,
garantindo que todo o pipeline analise exatamente o mesmo conjunto de edicoes e obras.
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .util import como_bool


def extrair_ano(serie: pd.Series) -> pd.Series:
    """Extrai o primeiro inteiro de 4 digitos de strings de data ('2023-05-10' -> 2023.0)."""
    return serie.astype(str).str.extract(r"(\d{4})")[0].astype(float)


def filtrar_escopo(df: pd.DataFrame, ano_min: int = 2015) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Aplica o filtro metodologico do projeto:
      1. Edicoes PT-BR com Ano_Publicacao_PT >= ano_min (edicoes sem ano parseavel sao excluidas).
      2. Exclusao de formatos fora de escopo (Flag_Formato_Excluir == True: manga, HQ, infantil).
    Devolve (df_filtrado, metadados_do_filtro).
    """
    n_inicial = len(df)
    meta: Dict[str, int] = {
        "ano_min": ano_min,
        "n_inicial": n_inicial,
        "n_sem_ano": 0,
        "n_antes_ano_min": 0,
        "n_formato_excluido": 0,
        "n_final": n_inicial,
    }
    if df.empty or "Ano_Publicacao_PT" not in df.columns:
        return df.copy(), meta

    d = df.copy()
    ano = extrair_ano(d["Ano_Publicacao_PT"])
    meta["n_sem_ano"] = int(ano.isna().sum())
    meta["n_antes_ano_min"] = int((ano < ano_min).sum())
    d["Ano"] = ano
    d = d[ano >= ano_min].copy()
    d["Ano"] = d["Ano"].astype(int)

    if "Flag_Formato_Excluir" in d.columns and not d.empty:
        excl = d["Flag_Formato_Excluir"].map(como_bool)
        meta["n_formato_excluido"] = int(excl.sum())
        d = d[~excl].copy()

    d = d.reset_index(drop=True)
    meta["n_final"] = len(d)
    return d, meta


def chave_obra(df: pd.DataFrame) -> pd.Series:
    """Identificador unico de Obra para agrupamento: Hardcover_Book_ID ou 'isbn:<ISBN_PT>' quando ausente."""
    tem_id = df["Hardcover_Book_ID"].notna() if "Hardcover_Book_ID" in df.columns else pd.Series(False, index=df.index)
    hc_str = (
        df["Hardcover_Book_ID"].astype(str).str.replace(r"\.0$", "", regex=True)
        if "Hardcover_Book_ID" in df.columns
        else pd.Series("", index=df.index)
    )
    isbn_str = "isbn:" + df["ISBN_PT"].astype(str) if "ISBN_PT" in df.columns else pd.Series(df.index.astype(str), index=df.index)
    return pd.Series(np.where(tem_id, hc_str, isbn_str), index=df.index)


def deduplicar_por_obra(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplica o DataFrame no nivel da Obra intelectual (Hardcover_Book_ID).
    Edicoes sem Hardcover_Book_ID (NaN) sao preservadas individualmente em vez de colapsarem numa unica linha.
    """
    if "Hardcover_Book_ID" not in df.columns or df.empty:
        return df.copy()
    sem_id = df[df["Hardcover_Book_ID"].isna()]
    com_id = df[df["Hardcover_Book_ID"].notna()].drop_duplicates("Hardcover_Book_ID")
    return pd.concat([com_id, sem_id]).sort_index()


def resolver_colunas_rotulo(df: pd.DataFrame) -> Tuple[str, str, str]:
    """
    Resolve os nomes das colunas do rotulo triangulado (v3.1 usa Sensivel_Label*;
    mantem fallback automatico para Spice_Label* em CSVs antigos da v3.0).
    """
    if "Sensivel_Label" in df.columns:
        return "Sensivel_Label", "Sensivel_Label_Confianca", "Sensivel_Label_Fontes"
    return "Spice_Label", "Spice_Label_Confianca", "Spice_Label_Fontes"
