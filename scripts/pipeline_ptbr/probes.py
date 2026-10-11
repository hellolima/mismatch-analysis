"""Diagnosticos: schema do Hardcover e viabilidade inversa (autora -> edicoes PT direto no Hardcover)."""
import os
import json

import pandas as pd

from ptbr_common import eh_isbn_br, so_digitos
from . import config as C
from .hardcover import hc


def inspecionar_schema():
    for tipo in ("editions", "books", "languages"):
        d = hc('query { __type(name: "%s") { fields { name type { name kind ofType { name kind } } } } }' % tipo)
        t = ((d or {}).get("data") or {}).get("__type")
        print(f"\n--- {tipo} ---")
        if not t:
            print(f"  sem introspeccao ({(d or {}).get('errors')}). Use o schema explorer do Hardcover.")
            continue
        for f in t["fields"]:
            ty = f["type"].get("name") or (f["type"].get("ofType") or {}).get("name")
            print(f"  {f['name']:<28} {ty}")


def sondar_inverso(autora: str, n: int = 50):
    """EXPERIMENTAL: os nomes de campo/relacao abaixo sao palpites; ajuste com a saida de 'schema'."""
    q = ('query { editions(where: {language: {language: {_eq: "Portuguese"}}, '
         'book: {contributions: {author: {name: {_eq: %s}}}}}, limit: %d) '
         '{ id isbn_13 title book_id } }' % (json.dumps(autora), n))
    d = hc(q)
    if not d or "errors" in d:
        print(f"Consulta inversa falhou: {(d or {}).get('errors')}\nRode '--sondar schema' e ajuste a query.")
        return
    eds = d["data"]["editions"]
    br = [e for e in eds if e.get("isbn_13") and eh_isbn_br(e["isbn_13"])]
    print(f"{autora}: {len(eds)} edicoes em portugues no Hardcover, {len(br)} com ISBN BR")
    for e in br[:15]:
        print(f"  {e['isbn_13']}  {e['title']}")
    if os.path.exists(C.SEED_CSV):
        seed = pd.read_csv(C.SEED_CSV, dtype=str)
        no_seed = set(seed["ISBN"].map(so_digitos))
        print(f"  ja presentes no seed Google Books: {sum(so_digitos(e['isbn_13']) in no_seed for e in br)}/{len(br)}")
