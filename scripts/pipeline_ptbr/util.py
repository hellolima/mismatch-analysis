"""Utilitarios de I/O atomico e sanitizacao de valores escalares."""
import os
import re
import sys
import json
from typing import List, Optional


def v(x) -> Optional[str]:
    """Converte NaN/''/'nan'/'None' para None; caso contrario devolve string aparada."""
    if x is None or (isinstance(x, float) and x != x):
        return None
    s = str(x).strip()
    return None if s in ("", "nan", "None") else s


def como_bool(x) -> bool:
    """Interpreta representacoes booleanas em CSV ('True', '1', '1.0', True)."""
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ("true", "1", "1.0")


def carregar_tabela(caminho: str, colunas: List[str], obrigatorio: bool = True) -> List[dict]:
    """Le listas delimitadas por '|' ou TAB. Linhas vazias ou iniciadas por '#' sao ignoradas."""
    if not os.path.exists(caminho):
        if obrigatorio:
            sys.exit(f"Arquivo obrigatorio nao encontrado: {caminho}")
        return []
    linhas = []
    with open(caminho, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            partes = [p.strip() for p in re.split(r"\s*\|\s*|\t", ln)]
            partes += [""] * (len(colunas) - len(partes))
            linhas.append(dict(zip(colunas, partes)))
    if obrigatorio and not linhas:
        sys.exit(f"O arquivo {caminho} existe mas esta vazio.")
    return linhas


def ler_json(caminho: str, padrao):
    if not os.path.exists(caminho):
        return padrao
    try:
        with open(caminho, encoding="utf-8") as f:
            return json.load(f)
    except ValueError:
        return padrao


def gravar_json(caminho: str, obj) -> None:
    """Escrita atomica via arquivo temporario: um Ctrl+C no meio nao corrompe o cache."""
    tmp = caminho + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, caminho)
