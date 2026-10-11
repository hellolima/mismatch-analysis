"""
run.py  --  PONTO DE ENTRADA UNICO do pipeline BookTok PT-BR.

    python scripts/run.py                 # roda tudo: seed -> ponte -> capas -> eda  (retomavel e incremental)
    python scripts/run.py --teste         # primeira vez: amostra pequena de seed + ponte (sem capas/eda)
    python scripts/run.py --teste --limite-autores 30 --limite-termos 10 --max-paginas 3 --max-req 300 --limite 300   # teste maior
    python scripts/run.py --etapas ponte,eda
    python scripts/run.py --sondar schema            # diagnostico do schema do Hardcover
    python scripts/run.py --sondar inverso --autora "Colleen Hoover"

Pode rodar quantas vezes quiser: o seed retoma do checkpoint (data/seeds/queries_done.jsonl), a ponte so consulta
ISBNs novos (data/cache/ponte_cache.json) e as capas so baixam as que faltam. Para refazer as falhas da ponte
(ex.: depois de instalar sentence-transformers): --retentar-falhas.
"""
import os
import sys
import logging
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_ptbr import config as C                       # noqa: E402
from pipeline_ptbr.util import carregar_tabela              # noqa: E402

ORDEM = ["seed", "ponte", "capas", "eda"]
log = logging.getLogger("run")


def etapa_seed(a):
    from pipeline_ptbr.gbooks import seed_google_books
    quais = {c.strip() for c in a.canais.split(",") if c.strip()}
    canais = {}
    if "autores" in quais:
        canais["autores"] = carregar_tabela(a.autores, ["nome", "estrato", "origem", "tipo_editorial"])[: a.limite_autores]
    if "termos" in quais:
        canais["termos"] = carregar_tabela(a.termos, ["termo", "subtipo"])[: a.limite_termos]
    if "editoras" in quais:
        canais["editoras"] = carregar_tabela(a.editoras, ["nome", "tipo"], obrigatorio=False)
    if "titulos" in quais:
        canais["titulos"] = carregar_tabela(a.titulos, ["titulo", "obs"], obrigatorio=False)
    log.info("Canais: " + ", ".join(f"{k}={len(x)}" for k, x in canais.items()))
    seed_google_books(canais, [o.strip() for o in a.ordens.split(",")], a.max_paginas,
                      a.max_req, a.subject, a.termo_editora)


def etapa_ponte(a):
    from pipeline_ptbr.gbooks import carregar_df_seed
    from pipeline_ptbr.bridge import executar_ponte
    df = carregar_df_seed()
    if df is None or df.empty:
        sys.exit("Seed vazio ou inexistente: rode a etapa 'seed' antes.")
    executar_ponte(df, a.limite, not a.sem_embeddings, a.min_count_spice, a.retentar_falhas)


def etapa_capas(a):
    from pipeline_ptbr.covers import baixar_capas
    baixar_capas(a.limite)


def etapa_eda(a):
    if not os.path.exists(C.OUT_LIVROS):
        log.warning(f"[eda] {C.OUT_LIVROS} nao existe; pulando")
        return
    log.info("[eda] executando 02_eda_booktok_ptbr.py")
    subprocess.run([sys.executable, C.EDA_SCRIPT], check=False)


def main():
    ap = argparse.ArgumentParser(description="Pipeline BookTok PT-BR: seed -> ponte -> capas -> eda")
    ap.add_argument("--etapas", default=None, help=f"subconjunto de {','.join(ORDEM)} (padrao: todas)")
    ap.add_argument("--teste", action="store_true",
                    help="amostra pequena (5 autoras, 3 termos, 2 paginas, 30 edicoes, 60 req); so seed+ponte. Qualquer flag --limite-*/--max-* que voce passar substitui o padrao do teste")
    ap.add_argument("--sondar", choices=["schema", "inverso"], default=None)
    ap.add_argument("--autora", default="Colleen Hoover", help="para --sondar inverso")
    # seed
    ap.add_argument("--autores", default=C.AUTORES_TXT)
    ap.add_argument("--termos", default=C.TERMOS_TXT)
    ap.add_argument("--editoras", default=C.EDITORAS_TXT)
    ap.add_argument("--titulos", default=C.TITULOS_TXT)
    ap.add_argument("--canais", default="autores,termos,editoras,titulos")
    ap.add_argument("--ordens", default="relevance,newest", help="orderBy para termos e editoras")
    ap.add_argument("--max-paginas", type=int, default=None, help="paginas por consulta (padrao 10; no --teste, 2)")
    ap.add_argument("--max-req", type=int, default=None, help="orcamento de req ao Google Books por execucao (padrao 900; no --teste, 60)")
    ap.add_argument("--subject", default=None, help="ex.: romance (valide antes)")
    ap.add_argument("--termo-editora", default="romance")
    ap.add_argument("--limite-autores", type=int, default=None)
    ap.add_argument("--limite-termos", type=int, default=None)
    # ponte / capas
    ap.add_argument("--limite", type=int, default=None, help="so N edicoes na ponte/capas")
    ap.add_argument("--sem-embeddings", action="store_true")
    ap.add_argument("--modelo-emb", default=C.MODELO_EMB)
    ap.add_argument("--min-count-spice", type=int, default=2)
    ap.add_argument("--retentar-falhas", action="store_true", help="refaz as edicoes que a ponte nao resolveu")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    C.garantir_dirs()
    C.MODELO_EMB = a.modelo_emb

    # padroes; no --teste sao menores, e qualquer flag explicita SEMPRE vence o padrao do teste
    padrao = {"limite_autores": 5, "limite_termos": 3, "max_paginas": 2, "limite": 30, "max_req": 60} if a.teste \
        else {"max_paginas": 10, "max_req": 900}
    for k, val in padrao.items():
        if getattr(a, k, None) is None:
            setattr(a, k, val)
    etapas = [e.strip() for e in (a.etapas or ("seed,ponte" if a.teste else ",".join(ORDEM))).split(",") if e.strip()]
    invalidas = [e for e in etapas if e not in ORDEM]
    if invalidas:
        sys.exit(f"Etapas invalidas: {invalidas}. Use: {ORDEM}")

    precisa_hc = a.sondar or "ponte" in etapas
    if precisa_hc and not C.HC_TOKEN:
        sys.exit("HARDCOVER_API_TOKEN ausente no .env")

    if a.sondar:
        from pipeline_ptbr.probes import inspecionar_schema, sondar_inverso
        inspecionar_schema() if a.sondar == "schema" else sondar_inverso(a.autora)
        return

    fns = {"seed": etapa_seed, "ponte": etapa_ponte, "capas": etapa_capas, "eda": etapa_eda}
    for e in ORDEM:
        if e in etapas:
            log.info(f"===== etapa: {e} =====")
            fns[e](a)


if __name__ == "__main__":
    main()