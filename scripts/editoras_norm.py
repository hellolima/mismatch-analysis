"""Chave canonica de editora (somente stdlib). Compartilhada pela ponte (bridge.py) e pela EDA.

A chave serve para AGRUPAR grafias do mesmo selo ('Arqueiro' / 'Editora Arqueiro' / 'Harlequin Brasil').
A coluna bruta Editora_PT nunca e alterada. Decisoes que regra nenhuma resolve (selo x grupo) entram em
ALIASES_EDITORA e devem ser registradas na metodologia.

Colocar em: scripts/editoras_norm.py (ao lado de ptbr_common.py).
"""
import re
import unicodedata


def _sem_acento(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


_PREF = re.compile(r"^(grupo editorial|editoras|editora|editorial|grupo|selo|ed\.)\s+")
_SUF = re.compile(r"\s+(editora|editores|editorial|ltda\.?|s\.?a\.?|eireli|do brasil|brasil|brazil)$")
_GENERICAS = ("editora", "editoras", "editorial", "editores", "grupo", "selo")

# Decisoes que a regra nao toma (chave -> chave de destino). Exemplos, desativados:
ALIASES_EDITORA = {
    # "gmt": "arqueiro",          # selo -> grupo (so se decidir agregar por grupo)
    # "globo livros": "globo",
}


def chave_editora(nome, aliases=None):
    """'Editora Arqueiro' -> 'arqueiro'; 'Harlequin Brasil' -> 'harlequin'; ausente/'nan' -> None."""
    if nome is None or str(nome).strip().lower() in ("", "nan", "none"):
        return None
    k = re.sub(r"[^\w\s\.]", " ", _sem_acento(str(nome)).lower())
    k = re.sub(r"\s+", " ", k).strip()
    prev = None
    while k != prev:                       # prefixos/sufixos em cascata
        prev = k
        k = _PREF.sub("", k)
        k = _SUF.sub("", k).strip()
    if k in _GENERICAS:                    # so havia a palavra generica
        return None
    k = (ALIASES_EDITORA if aliases is None else aliases).get(k, k)
    return k or None


def pares_suspeitos(chaves):
    """Pares (curta, longa) em que uma chave e prefixo, por palavra, da outra ('rocco' / 'rocco jovens leitores').
    Podem ser a mesma editora ou selos distintos: decisao humana, nao automatica."""
    ks = sorted({k for k in chaves if k})
    return [(a, b) for a in ks for b in ks if a != b and b.startswith(a + " ")]