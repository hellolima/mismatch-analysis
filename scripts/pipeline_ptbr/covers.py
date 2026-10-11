"""Etapa 3: baixa e versiona capas (URLs mudam; CLIP precisa >= 224 px)."""
import io
import os
import sys
import time
import hashlib
import logging
from datetime import datetime
from typing import Optional

import requests
import pandas as pd

from . import config as C
from .util import v

log = logging.getLogger("pipeline.covers")


def baixar_capas(limite: Optional[int] = None, lado_min: int = 200):
    try:
        from PIL import Image
    except ImportError:
        Image = None
        log.warning("[capas] Pillow ausente: sem checagem de dimensao nem conversao para JPEG (pip install pillow)")
    if not os.path.exists(C.OUT_EDICOES):
        log.warning(f"[capas] {C.OUT_EDICOES} nao existe (rode a ponte antes); pulando")
        return
    df = pd.read_csv(C.OUT_EDICOES, dtype=str)
    if limite:
        df = df.head(limite)
    manifest = pd.read_csv(C.CAPAS_MANIFEST, dtype=str).to_dict("records") if os.path.exists(C.CAPAS_MANIFEST) else []
    feitos = {m["ISBN_PT"] for m in manifest if v(m.get("Arquivo"))}
    for i, r in enumerate(df.to_dict("records")):
        isbn = r["ISBN_PT"]
        if isbn in feitos:
            continue
        cands = [("gbooks_hr", v(r.get("Capa_URL_PT_HR"))), ("hardcover_edicao", v(r.get("Capa_URL_Edicao_HC"))),
                 ("openlibrary", f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg?default=false"),
                 ("gbooks_thumb", v(r.get("Capa_URL_PT")))]
        achou = None
        for fonte, url in cands:
            if not url:
                continue
            try:
                resp = requests.get(url, timeout=(5, 20))
            except requests.RequestException:
                continue
            if resp.status_code != 200 or not resp.headers.get("content-type", "").startswith("image"):
                continue
            raw = resp.content
            w = h = None
            if Image is not None:
                try:
                    w, h = Image.open(io.BytesIO(raw)).size
                except Exception:
                    continue
                if min(w, h) < lado_min and fonte != "gbooks_thumb":
                    continue
            elif len(raw) < 5000:
                continue
            achou = (fonte, url, raw, w, h)
            break
        manifest = [m for m in manifest if m["ISBN_PT"] != isbn]
        if achou:
            fonte, url, raw, w, h = achou
            arq = os.path.join(C.COVERS_DIR, f"{isbn}.jpg")
            if Image is not None:
                Image.open(io.BytesIO(raw)).convert("RGB").save(arq, "JPEG", quality=95)
            else:
                with open(arq, "wb") as f:
                    f.write(raw)
            manifest.append({"ISBN_PT": isbn, "Arquivo": os.path.relpath(arq, C.ROOT), "Fonte": fonte, "URL": url,
                             "Largura": w, "Altura": h, "Bytes": len(raw),
                             "SHA256": hashlib.sha256(raw).hexdigest(), "Baixado_Em": datetime.now().isoformat()})
        else:
            manifest.append({"ISBN_PT": isbn, "Arquivo": "", "Fonte": "", "URL": "", "Largura": "", "Altura": "",
                             "Bytes": "", "SHA256": "", "Baixado_Em": datetime.now().isoformat()})
        if (i + 1) % 25 == 0:
            pd.DataFrame(manifest).to_csv(C.CAPAS_MANIFEST, index=False)
            log.info(f"[capas] {i + 1}/{len(df)}")
        time.sleep(0.2)
    m = pd.DataFrame(manifest)
    if m.empty:
        return
    # Google devolve a MESMA imagem 'capa indisponivel' para varios livros: hash repetido >= 3 = suspeito
    cont = m[m["SHA256"].fillna("") != ""]["SHA256"].value_counts()
    m["Placeholder_Suspeito"] = m["SHA256"].map(lambda h: bool(h) and h == h and cont.get(h, 0) >= 3)
    m.to_csv(C.CAPAS_MANIFEST, index=False)
    ok = int((m["Arquivo"].fillna("") != "").sum())
    log.info(f"[capas] {ok}/{len(m)} baixadas; placeholders suspeitos: {int(m['Placeholder_Suspeito'].sum())} "
             f"-> {C.CAPAS_MANIFEST}")
