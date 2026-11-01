"""
run.py  --  PONTO DE ENTRADA UNICO do pipeline BookTok PT-BR.

    python scripts/run.py                 # roda tudo: seed -> ponte -> rotulagem -> capas -> eda
    python scripts/run.py --teste         # primeira vez: amostra pequena de seed + ponte (sem capas/eda)
    python scripts/run.py --teste --limite-autores 30 --limite-termos 10 --max-paginas 3 --max-req 300 --limite 300
    python scripts/run.py --etapas ponte,rotulagem,eda
    python scripts/run.py --sondar schema            # diagnostico do schema do Hardcover
    python scripts/run.py --sondar inverso --autora "Colleen Hoover"
    python scripts/run.py --sondar rotulo            # diagnostico de cobertura e circularidade do rotulo

Pode rodar quantas vezes quiser: o seed retoma do checkpoint (data/seeds/queries_done.jsonl), a ponte so consulta
ISBNs novos (data/cache/ponte_cache.json) e as capas so baixam as que faltam. Para refazer as falhas da ponte
(ex.: depois de instalar sentence-transformers): --retentar-falhas.
"""
import os
import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_ptbr import config as C                       # noqa: E402
from pipeline_ptbr.util import carregar_tabela              # noqa: E402

ORDEM = ["seed", "ponte", "rotulagem", "capas", "eda"]
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


def etapa_rotulagem(a):
    if not os.path.exists(C.OUT_LIVROS):
        log.warning(f"[rotulagem] {C.OUT_LIVROS} nao existe (rode a ponte antes); pulando")
        return
    from pipeline_ptbr.rotulagem import executar_rotulagem
    from pipeline_ptbr.diagnostico_rotulo import executar_diagnostico_rotulo
    executar_rotulagem(Path(C.OUT_LIVROS), Path(C.OUT_LIVROS_ROTULADO), a.ano_min, a.pop_pct)
    executar_diagnostico_rotulo(Path(C.OUT_LIVROS_ROTULADO), a.ano_min)


def etapa_capas(a):
    from pipeline_ptbr.covers import baixar_capas
    baixar_capas(a.limite)


def etapa_eda(a):
    if not os.path.exists(C.OUT_LIVROS) and not os.path.exists(C.OUT_LIVROS_ROTULADO):
        log.warning(f"[eda] {C.OUT_LIVROS} nao existe; pulando")
        return
    import subprocess
    log.info("[eda] executando 02_eda_booktok_ptbr.py")
    subprocess.run([sys.executable, C.EDA_SCRIPT, "--ano-min", str(a.ano_min)], check=False)


def main():
    ap = argparse.ArgumentParser(description="Pipeline BookTok PT-BR: seed -> ponte -> rotulagem -> capas -> eda")
    ap.add_argument("--etapas", default=None, help=f"subconjunto de {','.join(ORDEM)} (padrao: todas)")
    ap.add_argument("--teste", action="store_true",
                    help="amostra pequena (5 autoras, 3 termos, 2 paginas, 30 edicoes, 60 req); so seed+ponte.")
    ap.add_argument("--sondar", choices=["schema", "inverso", "rotulo", "avaliacao", "gold"], default=None,
                    help="rotulo=diagnostico | avaliacao=ablacao/sensibilidade/kappa/funil | gold=planilhas e metricas humanas")
    ap.add_argument("--gold-acao", choices=["gerar", "avaliar"], default="avaliar", help="para --sondar gold")
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
    # ponte / capas / rotulagem / eda
    ap.add_argument("--limite", type=int, default=None, help="so N edicoes na ponte/capas")
    ap.add_argument("--sem-embeddings", action="store_true")
    ap.add_argument("--modelo-emb", default=C.MODELO_EMB)
    ap.add_argument("--min-count-spice", type=int, default=1,
                    help="votos minimos por tag p/ Flag_Spice_Tag E Spice_Nivel (mesmo limiar; padrao 1)")
    ap.add_argument("--retentar-falhas", action="store_true", help="refaz as edicoes que a ponte nao resolveu")
    ap.add_argument("--ano-min", type=int, default=C.ANO_MIN_PADRAO, help="ano minimo da edicao PT-BR (padrao 2015)")
    ap.add_argument("--pop-pct", type=float, default=C.POP_PCT_PADRAO, help="percentil de leitores p/ cobertura na rotulagem")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    C.garantir_dirs()
    C.MODELO_EMB = a.modelo_emb

    padrao = {"limite_autores": 5, "limite_termos": 3, "max_paginas": 2, "limite": 30, "max_req": 60} if a.teste \
        else {"max_paginas": 10, "max_req": 900}
    for k, val in padrao.items():
        if getattr(a, k, None) is None:
            setattr(a, k, val)
    etapas = [e.strip() for e in (a.etapas or ("seed,ponte" if a.teste else ",".join(ORDEM))).split(",") if e.strip()]
    invalidas = [e for e in etapas if e not in ORDEM]
    if invalidas:
        sys.exit(f"Etapas invalidas: {invalidas}. Use: {ORDEM}")

    tem_cache_ponte = os.path.exists(C.CACHE_PONTE) and os.path.exists(C.CACHE_OBRAS)
    precisa_hc = a.sondar in ("schema", "inverso") or ("ponte" in etapas and (not tem_cache_ponte or a.retentar_falhas))
    if precisa_hc and not C.HC_TOKEN:
        sys.exit("HARDCOVER_API_TOKEN ausente no .env")

    if a.sondar:
        if a.sondar == "rotulo":
            from pipeline_ptbr.diagnostico_rotulo import executar_diagnostico_rotulo
            executar_diagnostico_rotulo(Path(C.OUT_LIVROS_ROTULADO), a.ano_min)
        elif a.sondar == "avaliacao":
            from pipeline_ptbr.avaliar_rotulo import executar_avaliacao
            executar_avaliacao(a.ano_min, a.pop_pct)
        elif a.sondar == "gold":
            from pipeline_ptbr.gold_standard import gerar, avaliar
            gerar() if a.gold_acao == "gerar" else avaliar()
        else:
            from pipeline_ptbr.probes import inspecionar_schema, sondar_inverso
            inspecionar_schema() if a.sondar == "schema" else sondar_inverso(a.autora)
        return

    fns = {
        "seed": etapa_seed,
        "ponte": etapa_ponte,
        "rotulagem": etapa_rotulagem,
        "capas": etapa_capas,
        "eda": etapa_eda,
    }
    for e in ORDEM:
        if e in etapas:
            log.info(f"===== etapa: {e} =====")
            fns[e](a)


if __name__ == "__main__":
    main()