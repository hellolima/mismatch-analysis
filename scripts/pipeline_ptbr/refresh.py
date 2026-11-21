"""Etapa opcional `refresh`: re-consulta no Hardcover as obras JA resolvidas pela ponte.

Metricas (leitores, avaliacoes, nota), tags e content warnings mudam com o tempo; a ponte
ISBN -> obra nao. Esta etapa NAO refaz a ponte: so atualiza `obras_cache.json` para os IDs que
ja estao nele e reconstroi as tabelas processadas.

Reprodutibilidade:
  * Cada coleta vira um snapshot datado em data/snapshots/obras_hardcover_<AAAA-MM-DD>.json
    (versionado no git; data/cache/ nao e). O snapshot anterior e preservado antes de sobrescrever.
  * Cada obra recebe `_coletado_em`, que `montar_obra` exporta como `Coletado_Em_Hardcover`.
  * Um diff entre o snapshot anterior e o novo sai em outputs/tables/refresh_diff.csv e
    outputs/reports/refresh_resumo.txt (quantas obras ganharam CWs/tags, variacao de leitores etc.).
"""
import os
import json
import logging
from datetime import date, datetime
from typing import Dict

import pandas as pd

from . import config as C
from .util import ler_json, gravar_json
from .hardcover import buscar_obras
from .labeling import tags_por_categoria

log = logging.getLogger("pipeline.refresh")

SNAP_DIR = os.path.join(C.ROOT, "data", "snapshots")


def _caminho_snapshot(dia: str) -> str:
    return os.path.join(SNAP_DIR, f"obras_hardcover_{dia}.json")


def _resumo_obra(o: dict) -> dict:
    cats = tags_por_categoria(o.get("cached_tags"))
    return {
        "nota": o.get("rating"),
        "n_avaliacoes": o.get("ratings_count"),
        "n_leitores": o.get("users_count"),
        "n_tags": sum(len(v) for v in cats.values()),
        "n_cw": len(cats.get("Content Warning", [])),
        "cws": {t for t, _ in cats.get("Content Warning", [])},
    }


def _diff(antes: Dict[int, dict], depois: Dict[int, dict], dia_antes: str, dia_depois: str) -> pd.DataFrame:
    linhas = []
    for bid in sorted(set(antes) & set(depois)):
        a, d = _resumo_obra(antes[bid]), _resumo_obra(depois[bid])
        linhas.append({
            "Hardcover_Book_ID": bid, "Titulo": depois[bid].get("title"),
            "snapshot_antes": dia_antes, "snapshot_depois": dia_depois,
            "leitores_antes": a["n_leitores"], "leitores_depois": d["n_leitores"],
            "avaliacoes_antes": a["n_avaliacoes"], "avaliacoes_depois": d["n_avaliacoes"],
            "nota_antes": a["nota"], "nota_depois": d["nota"],
            "tags_antes": a["n_tags"], "tags_depois": d["n_tags"],
            "cw_antes": a["n_cw"], "cw_depois": d["n_cw"],
            "cw_novos": " | ".join(sorted(d["cws"] - a["cws"])),
            "cw_removidos": " | ".join(sorted(a["cws"] - d["cws"])),
        })
    return pd.DataFrame(linhas)


def _escrever_resumo(df: pd.DataFrame, n_falhas: int, caminho: str) -> None:
    def delta(col):
        return pd.to_numeric(df[f"{col}_depois"], errors="coerce") - pd.to_numeric(df[f"{col}_antes"], errors="coerce")
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("REFRESH DAS OBRAS (Hardcover)\n" + "=" * 60 + "\n")
        if df.empty:
            f.write("Sem obras em comum entre os snapshots.\n")
            return
        f.write(f"Snapshots: {df['snapshot_antes'].iloc[0]} -> {df['snapshot_depois'].iloc[0]}\n")
        f.write(f"Obras comparadas: {len(df)} | falhas na re-consulta (mantido valor antigo): {n_falhas}\n\n")
        for col, nome in (("leitores", "leitores"), ("avaliacoes", "avaliações"), ("tags", "tags (todas)"), ("cw", "content warnings")):
            dl = delta(col)
            f.write(f"{nome:<18} mudaram: {int((dl != 0).sum()):>4} obras | "
                    f"soma Δ: {int(dl.sum()):>6} | mediana Δ: {dl.median():.1f} | máx Δ: {int(dl.max())}\n")
        ganhou_cw = df[df["cw_novos"].ne("")]
        f.write(f"\nObras que ganharam algum content warning: {len(ganhou_cw)}\n")
        zero_antes = df[(pd.to_numeric(df["cw_antes"], errors="coerce") == 0) & (pd.to_numeric(df["cw_depois"], errors="coerce") > 0)]
        f.write(f"  das quais NÃO tinham nenhum CW antes: {len(zero_antes)}  (evidência direta de faltante não aleatório)\n")
        if len(ganhou_cw):
            f.write("\n  Top 15 por nº de CWs novos:\n")
            top = ganhou_cw.assign(_n=ganhou_cw["cw_novos"].str.count(r"\|") + 1).sort_values("_n", ascending=False).head(15)
            for _, r in top.iterrows():
                f.write(f"    - {str(r['Titulo'])[:50]:<50} +{r['_n']}: {r['cw_novos'][:90]}\n")


def executar_refresh(min_count_spice: int = 1) -> None:
    os.makedirs(SNAP_DIR, exist_ok=True)
    cache: Dict[int, dict] = {int(k): x for k, x in ler_json(C.CACHE_OBRAS, {}).items()}
    if not cache:
        raise SystemExit(f"{C.CACHE_OBRAS} vazio: rode a etapa 'ponte' antes.")

    # 1) preservar o snapshot anterior (data = _coletado_em se existir, senão mtime do cache)
    dia_antes = next((o.get("_coletado_em") for o in cache.values() if o.get("_coletado_em")), None) \
        or datetime.fromtimestamp(os.path.getmtime(C.CACHE_OBRAS)).date().isoformat()
    for o in cache.values():
        o.setdefault("_coletado_em", dia_antes)
    if not os.path.exists(_caminho_snapshot(dia_antes)):
        gravar_json(_caminho_snapshot(dia_antes), {str(k): v for k, v in cache.items()})
        log.info(f"[refresh] snapshot anterior preservado: {_caminho_snapshot(dia_antes)}")

    # 2) re-consultar todas as obras
    hoje = date.today().isoformat()
    ids = sorted(cache)
    log.info(f"[refresh] re-consultando {len(ids)} obras no Hardcover (~{len(ids) // 25 + 1} requisições)")
    novas = buscar_obras(ids)
    falhas = [i for i in ids if i not in novas]
    if falhas:
        log.warning(f"[refresh] {len(falhas)} obras não retornaram; mantendo valor anterior: {falhas[:10]}...")
    for bid, o in novas.items():
        o["_coletado_em"] = hoje
    depois = {**cache, **novas}

    # 3) diff + persistência (cache de trabalho e snapshot datado)
    df = _diff(cache, novas, dia_antes, hoje)
    C.garantir_dirs()
    df.to_csv(os.path.join(C.OUT_TABLES, "refresh_diff.csv"), index=False, encoding="utf-8")
    _escrever_resumo(df, len(falhas), os.path.join(C.OUT_REPORTS, "refresh_resumo.txt"))
    gravar_json(C.CACHE_OBRAS, {str(k): v for k, v in depois.items()})
    gravar_json(_caminho_snapshot(hoje), {str(k): v for k, v in depois.items()})
    log.info(f"[refresh] snapshot novo: {_caminho_snapshot(hoje)} | resumo: {C.OUT_REPORTS}/refresh_resumo.txt")

    # 4) reconstruir as tabelas processadas sem refazer a ponte (tudo já está em cache -> zero consultas)
    from .gbooks import carregar_df_seed
    from .bridge import executar_ponte
    seed = carregar_df_seed()
    if seed is None or seed.empty:
        log.warning("[refresh] seed ausente; tabelas processadas não foram reconstruídas")
        return
    executar_ponte(seed, usar_emb=False, min_count_spice=min_count_spice)


def usar_snapshot(dia: str, min_count_spice: int = 1) -> None:
    """Troca o cache de trabalho por um snapshot datado e reconstrói as tabelas (para reproduzir uma análise)."""
    p = _caminho_snapshot(dia)
    if not os.path.exists(p):
        raise SystemExit(f"Snapshot inexistente: {p}. Disponíveis: {sorted(os.listdir(SNAP_DIR)) if os.path.isdir(SNAP_DIR) else []}")
    gravar_json(C.CACHE_OBRAS, json.load(open(p, encoding="utf-8")))
    from .gbooks import carregar_df_seed
    from .bridge import executar_ponte
    executar_ponte(carregar_df_seed(), usar_emb=False, min_count_spice=min_count_spice)
    log.info(f"[refresh] tabelas reconstruídas a partir do snapshot {dia}")
