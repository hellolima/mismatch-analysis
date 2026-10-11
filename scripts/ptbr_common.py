"""
ptbr_common.py

Utilidades compartilhadas entre o seed/ponte (02_seed_ptbr_hardcover_v3.py) e o EDA
(02_eda_booktok_ptbr.py). Sem dependencias externas de proposito (so stdlib), para poder
ser importado por qualquer script.
"""
import re
import html
import difflib
import unicodedata
from typing import Iterable, List, Optional


# ---------------------------------------------------------------------------
# Normalizacao basica
# ---------------------------------------------------------------------------
def _vazio(s) -> bool:
    return s is None or (isinstance(s, float) and s != s)


def norm(s: Optional[str]) -> str:
    """minusculo, sem acento, sem pontuacao, espacos colapsados."""
    if _vazio(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def sem_acento(s: str) -> str:
    """minusculo e sem acento, MANTENDO pontuacao (usado nas regex de titulo)."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s.lower()).strip()


def sim(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    return difflib.SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def chunks(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ---------------------------------------------------------------------------
# ISBN
# ---------------------------------------------------------------------------
def so_digitos(isbn) -> str:
    return re.sub(r"[^\dX]", "", str(isbn or "").upper())


def eh_isbn_br(isbn: str) -> bool:
    return so_digitos(isbn).startswith(("97885", "97865"))


def isbn10_para_13(isbn10: str) -> Optional[str]:
    """ISBN-10 brasileiro (grupo 85) -> ISBN-13 (97885...)."""
    d = so_digitos(isbn10)
    if len(d) != 10 or not d.startswith("85"):
        return None
    stem = "978" + d[:9]
    soma = sum(int(c) * (1 if i % 2 == 0 else 3) for i, c in enumerate(stem))
    return f"{stem}{(10 - soma % 10) % 10}"


def isbn13_para_10(isbn13: str) -> Optional[str]:
    """ISBN-13 (978...) -> ISBN-10. Necessario porque parte das edicoes no Hardcover so tem isbn_10."""
    d = so_digitos(isbn13)
    if len(d) != 13 or not d.startswith("978") or "X" in d:
        return None
    core = d[3:12]
    soma = sum(int(c) * (10 - i) for i, c in enumerate(core))
    chk = (11 - soma % 11) % 11
    return core + ("X" if chk == 10 else str(chk))


# ---------------------------------------------------------------------------
# Pacotes / HTML
# ---------------------------------------------------------------------------
RE_KIT_BOX = re.compile(r"\b(kit|box|colecao)\b")


def eh_kit_ou_box(titulo: Optional[str]) -> bool:
    return bool(RE_KIT_BOX.search(norm(titulo)))


_RE_QUEBRA = re.compile(r"(?i)<br\s*/?>|</p>|</li>|</div>")
_RE_TAG = re.compile(r"<[^>]+>")


def limpar_html(s: Optional[str]) -> Optional[str]:
    """Remove tags HTML (<p>, <br>, <b>...) e entidades das descricoes do Google Books."""
    if _vazio(s):
        return None
    s = _RE_QUEBRA.sub(" ", str(s))
    s = _RE_TAG.sub(" ", s)
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


# ---------------------------------------------------------------------------
# Casamento de autora
# ---------------------------------------------------------------------------
def tokens_autor(s: str) -> List[str]:
    """Tokens normalizados; iniciais consecutivas viram um token so ('J. M.' e 'JM' -> 'jm')."""
    out, buf = [], ""
    for t in norm(s).split():
        if len(t) == 1:
            buf += t
        else:
            if buf:
                out.append(buf)
                buf = ""
            out.append(t)
    if buf:
        out.append(buf)
    return out


def autor_confere(consulta: str, autores_vol: Iterable[str]) -> bool:
    """
    Sobrenome obrigatorio. Primeiro nome obrigatorio se nao for inicial; se for inicial,
    exige todas as iniciais. Resultado:
      'Christina Lauren' NAO aceita 'Lauren Asher';  'A.C. Meyer' NAO aceita 'Stephenie Meyer';
      'Jennifer L. Armentrout' aceita 'Jennifer Armentrout' e 'Jennifer L. Armentrout'.
    """
    q = tokens_autor(consulta)
    autores_vol = list(autores_vol or [])
    if len(q) < 2:
        return any(norm(consulta) and norm(consulta) in norm(x) for x in autores_vol)
    sob, resto = q[-1], q[:-1]
    exigidos = [resto[0]] if len(resto[0]) > 1 else resto
    for a in autores_vol:
        t = tokens_autor(a)
        if sob in t and all(x in t for x in exigidos):
            return True
    return False


# ---------------------------------------------------------------------------
# Limpeza de titulo (movida do EDA: agora usada tambem na ponte)
# ---------------------------------------------------------------------------
RE_MARKETING = re.compile(
    r"sucesso|best.?seller|mais vendid|tiktok|booktok|"
    r"romance|comedia romantica|romantasi\w*|fantasia romantica|\bcozy\b|releitura|dark romance|"
    r"autor(a)? (de|do|da)\b|do autor|da autora|"
    r"nova edicao|edicao|capitulo extra|inedit|formula|especial|ilustrad|capa dura|"
    r"\b(v|vol|volume|livro|book)\.? ?\d+|trilogia|serie\b|colecao|\bbox\b|\bkit\b"
)
RE_SEP = re.compile(r"\s+[\u2013\u2014-]\s+|\s*:\s+")
RE_GRUPO = re.compile(r"\s*[\(\[]([^\)\]]*)[\)\]]")
RE_VOL_FINAL = re.compile(r"[,;]?\s*\b(v|vol|volume|livro)\.?\s*\d+\s*$", re.I)


def _parece_marketing(trecho: str) -> bool:
    return bool(RE_MARKETING.search(sem_acento(trecho)))


def _eh_titulo_original(trecho: str, titulo_original: Optional[str]) -> bool:
    if _vazio(titulo_original) or not titulo_original:
        return False
    a, b = sem_acento(trecho), sem_acento(str(titulo_original))
    return bool(a) and bool(b) and difflib.SequenceMatcher(None, a, b).ratio() >= 0.85


def _descartavel(trecho: str, titulo_original: Optional[str]) -> bool:
    return _parece_marketing(trecho) or _eh_titulo_original(trecho, titulo_original)


def limpar_titulo(titulo: Optional[str], titulo_original: Optional[str] = None) -> Optional[str]:
    """Mantem so o titulo da obra (remove marketing, edicao, serie e titulo original repetido)."""
    if _vazio(titulo):
        return titulo
    t = str(titulo).strip()
    if not t:
        return t
    t = RE_GRUPO.sub(lambda m: "" if _descartavel(m.group(1), titulo_original) else m.group(0), t)
    for m in RE_SEP.finditer(t):
        if _descartavel(t[m.end():], titulo_original) or RE_VOL_FINAL.search(t[:m.start()]):
            t = t[:m.start()]
            break
    t = RE_VOL_FINAL.sub("", t)
    t = re.sub(r"\s+", " ", t).strip(" \t-\u2013\u2014:,;.")
    return t or str(titulo).strip()