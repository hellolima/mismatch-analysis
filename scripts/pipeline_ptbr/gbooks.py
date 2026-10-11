"""Etapa 1: seed de edicoes brasileiras via Google Books (autoras, termos, editoras, titulos)."""
import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd

from ptbr_common import so_digitos, eh_isbn_br, isbn10_para_13, eh_kit_ou_box, limpar_html, autor_confere
from . import config as C
from .util import v

log = logging.getLogger("pipeline.gbooks")


class Cota(Exception):
    """Cota/orcamento do Google Books esgotado."""


class GB:
    def __init__(self, max_req: int):
        self.n = 0
        self.max = max_req

    def get(self, params: dict) -> Optional[dict]:
        if C.GB_KEY:
            params = {**params, "key": C.GB_KEY}
        ja_429 = False
        for t in range(3):
            if self.n >= self.max:
                raise Cota(f"orcamento de {self.max} requisicoes atingido")
            self.n += 1
            try:
                r = requests.get(C.GB_URL, params=params, timeout=(5, 15))
            except requests.RequestException as e:
                log.warning(f"rede ({type(e).__name__}); tentativa {t + 1}/3")
                time.sleep(2 * (t + 1))
                continue
            if r.status_code == 429:
                if ja_429:
                    raise Cota("429 persistente (cota diaria)")
                ja_429 = True
                log.warning("429: aguardando 30s antes de reconfirmar...")
                time.sleep(30)
                continue
            if r.status_code == 200:
                time.sleep(0.3)
                return r.json()
            if r.status_code >= 500:
                time.sleep(3)
                continue
            return None
        return None


class Estado:
    """Checkpoint por consulta: key -> {next_page, complete}. Append-only em JSONL."""

    def __init__(self, caminho: str):
        self.caminho = caminho
        self.d: Dict[str, dict] = {}
        if os.path.exists(caminho):
            with open(caminho, encoding="utf-8") as f:
                for ln in f:
                    try:
                        r = json.loads(ln)
                        self.d[r["key"]] = r
                    except (ValueError, KeyError):
                        continue

    def get(self, key: str) -> dict:
        return self.d.get(key, {"next_page": 0, "complete": False})

    def set(self, key: str, next_page: int, complete: bool, motivo: str = ""):
        r = {"key": key, "next_page": next_page, "complete": complete, "motivo": motivo,
             "ts": datetime.now().isoformat(timespec="seconds")}
        self.d[key] = r
        with open(self.caminho, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


SET_COLS = ["Consultas_Origem", "Tipos_Origem", "Estratos_Origem", "Origens_Pais"]


def _set(s) -> set:
    return {p.strip() for p in str(s).split(" | ") if p.strip()} if v(s) else set()


def _caminho_seed() -> Optional[str]:
    for c in (C.SEED_CSV, C.SEED_V2_CSV):
        if os.path.exists(c) and os.path.getsize(c) > 5:
            return c
    return None


def carregar_df_seed() -> Optional[pd.DataFrame]:
    """Seed como DataFrame (o v3; se nao existir, o v2 antigo)."""
    c = _caminho_seed()
    return pd.read_csv(c, dtype=str) if c else None


def carregar_seed_existente() -> Dict[str, dict]:
    caminho = _caminho_seed()
    rows: Dict[str, dict] = {}
    if not caminho:
        return rows
    try:
        df = pd.read_csv(caminho, dtype=str)
    except Exception as e:
        log.warning(f"[seed] nao foi possivel ler {caminho}: {e}")
        return rows
    for reg in df.to_dict("records"):
        reg = {k: v(x) for k, x in reg.items()}
        isbn = so_digitos(reg.get("ISBN"))
        if not isbn:
            continue
        if "Consulta_Origem" in reg:                    # migracao do schema v2
            reg["Consultas_Origem"] = reg.pop("Consulta_Origem")
        if "Tipo_Origem" in reg:
            reg["Tipos_Origem"] = reg.pop("Tipo_Origem")
        for c in SET_COLS:
            reg[c] = _set(reg.get(c))
        reg["ISBN"] = isbn
        rows[isbn] = reg
    log.info(f"[seed] {len(rows)} registros carregados de {os.path.basename(caminho)}")
    return rows


def salvar_seed(rows: Dict[str, dict]):
    df = pd.DataFrame([{k: (" | ".join(sorted(x)) if isinstance(x, set) else x) for k, x in r.items()}
                       for r in rows.values()])
    df.to_csv(C.SEED_CSV, index=False, encoding="utf-8")


def mesclar(rows: Dict[str, dict], isbn: str, reg: dict) -> bool:
    """Insere ou acumula procedencia. True se o ISBN e novo."""
    ex = rows.get(isbn)
    if ex is None:
        rows[isbn] = reg
        return True
    for c in SET_COLS:
        ex[c] = ex.get(c, set()) | reg[c]
    for k, val in reg.items():
        if k in SET_COLS:
            continue
        if ex.get(k) in (None, "") and val not in (None, ""):
            ex[k] = val
    return False


def processar_item(it: dict, ctx: dict) -> Optional[Tuple[str, dict, List[str]]]:
    vi = it.get("volumeInfo", {}) or {}
    if eh_kit_ou_box(vi.get("title")):
        return None
    ids = vi.get("industryIdentifiers", []) or []
    isbn13 = next((so_digitos(i.get("identifier")) for i in ids if i.get("type") == "ISBN_13"), None)
    if not isbn13:
        i10 = next((i.get("identifier") for i in ids if i.get("type") == "ISBN_10"), None)
        if i10:
            isbn13 = isbn10_para_13(i10)
    if not isbn13 or not eh_isbn_br(isbn13):
        return None

    autores_vol = vi.get("authors", []) or []
    gbid = it.get("id")
    capa = (vi.get("imageLinks", {}) or {}).get("thumbnail")
    tipo = ctx["canal"] + (f":{ctx['subtipo']}" if ctx.get("subtipo") else "")
    reg = {
        "ISBN": isbn13,
        "GB_ID": gbid,
        "Titulo": vi.get("title"),
        "Autor": ", ".join(autores_vol),
        "Editora": vi.get("publisher"),
        "Ano": vi.get("publishedDate"),
        "Capa_URL": capa.replace("http://", "https://") if capa else None,
        # resolucao maior para CLIP (o thumbnail tem ~128 px); validado na etapa 'capas'
        "Capa_URL_HR": (f"https://books.google.com/books/content?id={gbid}&printsec=frontcover&img=1&zoom=3"
                        if gbid else None),
        "Sinopse": limpar_html(vi.get("description")),
        "Categorias_GB": " | ".join(vi.get("categories", []) or []) or None,
        "Paginas_GB": vi.get("pageCount") or None,
        "Maturity_GB": vi.get("maturityRating"),
        "Idioma_GB": vi.get("language"),
        "E_Ebook": str((it.get("saleInfo") or {}).get("isEbook")),
        "Consultas_Origem": {ctx["consulta"]},
        "Tipos_Origem": {tipo},
        "Estratos_Origem": {ctx["estrato"]} if ctx.get("estrato") else set(),
        "Origens_Pais": {ctx["origem"]} if ctx.get("origem") else set(),
    }
    return isbn13, reg, autores_vol


def executar_consulta(gb: GB, estado: Estado, rows: Dict[str, dict], canal: str, q: str, ordem: str,
                      ctx: dict, filtro_autor: Optional[str], ate_pagina: int, max_final: int) -> int:
    key = f"{canal}|{q}|{ordem}"
    st = estado.get(key)
    if st["complete"]:
        return 0
    p = st["next_page"]
    novos_total = 0
    sem_novidade = 0
    while p < ate_pagina:
        params = {"q": q, "langRestrict": "pt", "maxResults": 40, "startIndex": p * 40,
                  "printType": "books", "orderBy": ordem}
        d = gb.get(params)                       # Cota sobe daqui; estado ja reflete a pagina anterior
        if d is None:
            estado.set(key, p, False, "erro_http")
            return novos_total
        items = d.get("items", []) or []
        total = int(d.get("totalItems", 0) or 0)
        validos = novos = 0
        for it in items:
            res = processar_item(it, ctx)
            if not res:
                continue
            isbn, reg, autores_vol = res
            if filtro_autor and not autor_confere(filtro_autor, autores_vol):
                continue
            validos += 1
            novos += int(mesclar(rows, isbn, reg))
        novos_total += novos
        p += 1
        sem_novidade = sem_novidade + 1 if novos == 0 else 0
        motivo = None
        if not items or p * 40 >= total:
            motivo = "fim_resultados"
        elif p == 1 and validos == 0:
            motivo = "sondagem_zero"             # nenhuma edicao BR valida na 1a pagina: nao aprofunda
        elif sem_novidade >= 2:
            motivo = "saturou"
        elif p >= max_final:
            motivo = "max_paginas"
        estado.set(key, p, motivo is not None, motivo or "")
        if motivo:
            break
    if novos_total:
        log.info(f"[{canal}] {q[:60]} ({ordem}) +{novos_total} (acumulado {len(rows)})")
    return novos_total


def seed_google_books(canais: Dict[str, List[dict]], ordens: List[str], max_paginas: int,
                      max_req: int, subject: Optional[str], termo_editora: str) -> int:
    """Roda o seed (retomavel). Devolve o numero de edicoes no seed."""
    rows = carregar_seed_existente()
    estado = Estado(C.STATE_JSONL)
    gb = GB(max_req)

    tarefas = []     # (canal, query, ordem, ctx, filtro_autor)
    for a in canais.get("autores", []):
        tarefas.append(("autora", f'inauthor:"{a["nome"]}"', "relevance",
                        {"canal": "autora", "consulta": a["nome"], "estrato": a.get("estrato"),
                         "origem": a.get("origem")}, a["nome"]))
    for t in canais.get("termos", []):
        for o in ordens:
            q = f'"{t["termo"]}"' + (f" subject:{subject}" if subject else "")
            tarefas.append(("termo", q, o, {"canal": "termo", "subtipo": t.get("subtipo"),
                                            "consulta": t["termo"]}, None))
    for e in canais.get("editoras", []):
        for o in ordens:
            q = f'inpublisher:"{e["nome"]}" {termo_editora}'.strip()
            tarefas.append(("editora", q, o, {"canal": "editora", "subtipo": e.get("tipo"),
                                              "consulta": e["nome"]}, None))
    for t in canais.get("titulos", []):
        tarefas.append(("titulo", f'intitle:"{t["titulo"]}"', "relevance",
                        {"canal": "titulo", "consulta": t["titulo"]}, None))
    log.info(f"[seed] {len(tarefas)} consultas planejadas; orcamento {max_req} req; max {max_paginas} pag/consulta")

    try:
        # Passo 1 = sondagem (1 pagina de TODAS as consultas); passo 2 = aprofunda so quem rendeu.
        for ate in sorted({1, max_paginas}):
            for i, (canal, q, ordem, ctx, filtro) in enumerate(tarefas):
                executar_consulta(gb, estado, rows, canal, q, ordem, ctx, filtro, ate, max_paginas)
                if (i + 1) % 10 == 0:
                    salvar_seed(rows)
            salvar_seed(rows)
            log.info(f"[seed] passo ate_pagina={ate} concluido: {len(rows)} edicoes, {gb.n} req")
    except Cota as e:
        log.error(f"[seed] interrompido ({e}). Checkpoint salvo; rode de novo para retomar.")
    finally:
        salvar_seed(rows)
    pend = sum(1 for (c, q, o, *_) in tarefas if not estado.get(f"{c}|{q}|{o}")["complete"])
    log.info(f"[seed] salvo em {C.SEED_CSV}: {len(rows)} edicoes; consultas incompletas: {pend}")
    return len(rows)
