"""Cliente Hardcover e estrategias de ponte: ISBN -> catalogo da autora (+embedding) -> busca."""
import json
import time
import logging
from typing import Dict, List, Optional, Tuple

import requests

from ptbr_common import norm, sim, chunks, isbn13_para_10, limpar_titulo, autor_confere
from . import config as C
from .util import ler_json, gravar_json

log = logging.getLogger("pipeline.hardcover")


# ---------------------------------------------------------------------------
# Cliente (plano free: 60 req/min -> pausa de 1.1s)
# ---------------------------------------------------------------------------
def hc(query: str, tentativas: int = 4) -> Optional[dict]:
    auth = C.HC_TOKEN if C.HC_TOKEN.startswith("Bearer ") else f"Bearer {C.HC_TOKEN}"
    headers = {"Content-Type": "application/json", "Authorization": auth}
    for t in range(tentativas):
        try:
            r = requests.post(C.HC_URL, json={"query": query}, headers=headers, timeout=(5, 30))
        except requests.RequestException as e:
            log.warning(f"Rede ({type(e).__name__}); tentativa {t + 1}/{tentativas}")
            time.sleep(3 * (t + 1))
            continue
        if r.status_code == 429:
            log.warning("429 (limite). Aguardando 20s...")
            time.sleep(20)
            continue
        if r.status_code in (401, 403):
            log.error(f"HTTP {r.status_code}: token invalido/expirado ou query bloqueada.")
            return None
        if r.status_code != 200:
            log.warning(f"HTTP {r.status_code}: {r.text[:200]}")
            time.sleep(3)
            continue
        time.sleep(1.1)
        try:
            return r.json()
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Ponte 1: ISBN -> edicao + obra numa unica query (com degradacao de schema)
# ---------------------------------------------------------------------------
# Variantes da query da obra, da mais rica para a mais simples; o codigo usa a primeira que o schema aceitar.
# `contribution` e o PAPEL (null = autor; "Translator", "Illustrator"...): sem ele, o tradutor aparece como autor.
BOOK_FIELDS = [
    "id title slug rating ratings_count release_year description users_count "
    "image { url } contributions { contribution author { name } } cached_tags",
    "id title slug rating ratings_count release_year description users_count "
    "image { url } contributions { author { name } } cached_tags",
    "id title rating ratings_count release_year description "
    "image { url } contributions { author { name } } cached_tags",
]
_ED_FULL = "id isbn_13 isbn_10 title pages release_date book_id image { url }"
VARIANTES_EDITIONS = [                 # (campos da edicao, campos da obra aninhada ou None)
    (_ED_FULL, BOOK_FIELDS[0]),
    (_ED_FULL, BOOK_FIELDS[1]),
    ("id isbn_13 isbn_10 title book_id", BOOK_FIELDS[2]),
    ("id isbn_13 title book_id", None),
]
_var_ok: Optional[int] = None
_campos_ok: Optional[int] = None


def consultar_editions(campo: str, valores: List[str]) -> Optional[List[dict]]:
    global _var_ok
    lista = ",".join(f'"{x}"' for x in valores)
    for k in ([_var_ok] if _var_ok is not None else range(len(VARIANTES_EDITIONS))):
        ef, bf = VARIANTES_EDITIONS[k]
        aninhado = f" book {{ {bf} }}" if bf else ""
        q = f"query {{ editions(where: {{{campo}: {{_in: [{lista}]}}}}, limit: 100) {{ {ef}{aninhado} }} }}"
        d = hc(q)
        if d and "errors" not in d:
            _var_ok = k
            return d["data"]["editions"]
        log.warning(f"editions (variante #{k}) falhou: {(d or {}).get('errors')}")
    return None


def ponte_por_isbn(isbns13: List[str]) -> Tuple[Dict[str, dict], Dict[int, dict]]:
    """Devolve ({isbn13: resolucao}, {book_id: obra}). A edicao guardada NAO inclui a obra aninhada."""
    res: Dict[str, dict] = {}
    obras_inline: Dict[int, dict] = {}

    def processar(campo: str, valores: List[str], para13: Optional[Dict[str, str]], metodo: str):
        for ch in chunks(valores, 25):
            eds = consultar_editions(campo, ch)
            for e in eds or []:
                if not e.get("book_id"):
                    continue
                k = e.get(campo)
                k13 = k if campo == "isbn_13" else (para13 or {}).get(k)
                if not k13 or k13 in res:
                    continue
                ed = {kk: vv for kk, vv in e.items() if kk != "book"}
                res[k13] = {"book_id": int(e["book_id"]), "metodo": metodo, "score": 1.0, "ed": ed}
                b = e.get("book")
                if b and b.get("id"):
                    obras_inline[int(b["id"])] = b

    processar("isbn_13", isbns13, None, "isbn13_edicao")
    pend = [i for i in isbns13 if i not in res]
    mapa10 = {isbn13_para_10(i): i for i in pend if isbn13_para_10(i)}
    processar("isbn_10", list(mapa10), mapa10, "isbn10_edicao")
    return res, obras_inline


def buscar_obras(ids: List[int]) -> Dict[int, dict]:
    global _campos_ok
    obras: Dict[int, dict] = {}
    for ch in chunks(sorted(set(ids)), 25):
        lista = ",".join(str(i) for i in ch)
        for k in ([_campos_ok] if _campos_ok is not None else range(len(BOOK_FIELDS))):
            q = (f"query {{ books(where: {{id: {{_in: [{lista}]}}}}, limit: 25) "
                 f"{{ {BOOK_FIELDS[k]} }} }}")
            d = hc(q)
            if d and "errors" not in d:
                _campos_ok = k
                for b in d["data"]["books"]:
                    obras[int(b["id"])] = b
                break
            log.warning(f"books (campos #{k}) falhou: {(d or {}).get('errors')}")
    return obras


# ---------------------------------------------------------------------------
# Ponte 2: catalogo da autora (Hardcover) + embedding multilingue
# ---------------------------------------------------------------------------
_cache_catalogo: Optional[Dict[str, list]] = None


def catalogo_autora(nome: str) -> List[dict]:
    """Obras da autora no Hardcover via search (filtradas por autor_confere). Cache em disco."""
    global _cache_catalogo
    if _cache_catalogo is None:
        _cache_catalogo = ler_json(C.CACHE_CATALOGO, {})
    chave = norm(nome)
    if chave in _cache_catalogo:
        return _cache_catalogo[chave]
    docs: Dict[int, dict] = {}
    for pagina in (1, 2, 3):
        q = ('query { search(query: %s, query_type: "Book", per_page: 25, page: %d) { results } }'
             % (json.dumps(nome), pagina))
        d = hc(q)
        try:
            hits = d["data"]["search"]["results"]["hits"]
        except (TypeError, KeyError):
            break
        if not hits:
            break
        for h in hits:
            doc = h.get("document", {}) or {}
            if doc.get("id") and autor_confere(nome, doc.get("author_names") or []):
                docs[int(doc["id"])] = {"id": int(doc["id"]), "title": doc.get("title"),
                                        "alternative_titles": list(doc.get("alternative_titles") or []),
                                        "author_names": doc.get("author_names") or []}
        if len(hits) < 25:
            break
    _cache_catalogo[chave] = list(docs.values())
    gravar_json(C.CACHE_CATALOGO, _cache_catalogo)
    return _cache_catalogo[chave]


class Embedder:
    def __init__(self, modelo: str):
        from sentence_transformers import SentenceTransformer     # import tardio: dependencia opcional
        self.m = SentenceTransformer(modelo)

    def enc(self, textos: List[str]):
        return self.m.encode(textos, normalize_embeddings=True, show_progress_bar=False)


def casar_catalogo(titulo: str, autor: str, embedder: Optional[Embedder]) -> Optional[dict]:
    primeiro = (autor or "").split(",")[0].strip()
    if not primeiro or not titulo:
        return None
    docs = catalogo_autora(primeiro)
    if not docs:
        return None
    tit = limpar_titulo(titulo) or titulo
    nomes, donos = [], []
    for d in docs:
        for n in [d.get("title")] + list(d.get("alternative_titles") or []):
            if n:
                nomes.append(n)
                donos.append(d["id"])
    if not nomes:
        return None
    fuzzy: Dict[int, float] = {}
    for n, o in zip(nomes, donos):
        fuzzy[o] = max(fuzzy.get(o, 0.0), sim(tit, n))
    emb: Dict[int, float] = {}
    if embedder is not None:
        s = embedder.enc(nomes) @ embedder.enc([tit])[0]
        for o, x in zip(donos, s):
            emb[o] = max(emb.get(o, -1.0), float(x))
    comb = {o: max(fuzzy[o], emb.get(o, 0.0)) for o in fuzzy}
    rank = sorted(comb.items(), key=lambda kv: -kv[1])
    best, sc = rank[0]
    margem = sc - (rank[1][1] if len(rank) > 1 else 0.0)
    aceito = fuzzy[best] >= C.FUZZY_ACEITE or (emb.get(best, 0.0) >= C.EMB_ACEITE and margem >= C.EMB_MARGEM)
    if not aceito:
        return None
    return {"book_id": best, "metodo": "catalogo_autora", "score": round(sc, 3),
            "fuzzy": round(fuzzy[best], 3), "emb": round(emb[best], 3) if best in emb else None,
            "margem": round(margem, 3)}


def ponte_por_busca(titulo: str, autor: str) -> Optional[dict]:
    """Fallback final: search titulo(limpo)+autor, fuzzy >= 0.90 e autor conferindo."""
    tit = limpar_titulo(titulo) or titulo
    primeiro = (autor or "").split(",")[0].strip()
    termo = f"{tit} {primeiro}".strip()
    q = ('query { search(query: %s, query_type: "Book", per_page: 5, page: 1) { results } }'
         % json.dumps(termo))
    d = hc(q)
    try:
        hits = d["data"]["search"]["results"]["hits"]
    except (TypeError, KeyError):
        return None
    melhor = None
    for h in hits:
        doc = h.get("document", {}) or {}
        nomes = [doc.get("title")] + list(doc.get("alternative_titles") or [])
        score = max((sim(tit, n) for n in nomes if n), default=0.0)
        autor_ok = (not primeiro) or (not doc.get("author_names")) or autor_confere(primeiro, doc["author_names"])
        if score >= C.FUZZY_ACEITE and autor_ok and doc.get("id"):
            if melhor is None or score > melhor["score"]:
                melhor = {"book_id": int(doc["id"]), "metodo": "busca_titulo_autor",
                          "score": round(score, 3), "fuzzy": round(score, 3), "emb": None, "margem": None}
    return melhor