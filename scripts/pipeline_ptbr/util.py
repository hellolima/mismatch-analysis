import os
import re
import sys
import json
from typing import List, Optional


def v(x) -> Optional[str]:
    """NaN/''/'nan'/'None' -> None; senao string aparada."""
    if x is None or (isinstance(x, float) and x != x):
        return None
    s = str(x).strip()
    return None if s in ("", "nan", "None") else s


def carregar_tabela(caminho: str, colunas: List[str], obrigatorio: bool = True) -> List[dict]:
    """Le listas 'a | b | c' (ou TAB). Linhas com '#' sao comentarios."""
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
    """Escrita atomica: um Ctrl+C no meio nao corrompe o cache."""
    tmp = caminho + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, caminho)
