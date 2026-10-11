"""Etapa 2: ponte edicao PT -> obra (Hardcover). INCREMENTAL: so consulta ISBNs que ainda nao estao no cache."""
import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ptbr_common import norm, so_digitos, eh_kit_ou_box, limpar_titulo, autor_confere
from . import config as C
from .util import v, carregar_tabela, ler_json, gravar_json
from .hardcover import ponte_por_isbn, buscar_obras, casar_catalogo, ponte_por_busca, Embedder
from .labeling import montar_obra

log = logging.getLogger("pipeline.bridge")


def carregar_tipos_editora() -> List[Tuple[str, str]]:
    return [(norm(r["nome"]), r["tipo"]) for r in carregar_tabela(C.EDITORAS_TXT, ["nome", "tipo"], False)
            if norm(r["nome"])]


def tipo_editorial(editora: Optional[str], tipos: List[Tuple[str, str]]) -> Optional[str]:
    if not editora:
        return None
    n = f" {norm(editora)} "          # palavra inteira: 'alt' nao pode casar com 'Alta Novel'
    for nome, tipo in tipos:
        if nome and f" {nome} " in n:
            return tipo
    return "desconhecido"


def executar_ponte(df_seed: pd.DataFrame, limite: Optional[int] = None, usar_emb: bool = True,
                   min_count_spice: int = 2, retentar_falhas: bool = False):
    """
    Cache da ponte (data/cache/ponte_cache.json): isbn -> {"res": resolucao|None, "emb": bool|None}
      res != None            resolvido (nunca refeito)
      res None, emb None     so o ISBN foi tentado; falta o fallback
      res None, emb False    fallback tentado SEM embeddings; e refeito se os embeddings forem ativados
      res None, emb True     fallback completo tentado; so refeito com --retentar-falhas
    """
    df_seed = df_seed.copy()
    if limite and limite < len(df_seed):
        # amostra aleatoria reprodutivel (o head() so pegaria as primeiras autoras da lista)
        df_seed = df_seed.sample(n=limite, random_state=42)
    eh_pacote = df_seed["Titulo"].map(eh_kit_ou_box)
    if eh_pacote.any():
        log.info(f"[ponte] descartando {int(eh_pacote.sum())} edicoes Kit/Box")
    df_seed = df_seed[~eh_pacote]
    df_seed["ISBN"] = df_seed["ISBN"].astype(str).str.replace(r"\.0$", "", regex=True)
    seeds = [{k: v(x) for k, x in r.items()} for r in df_seed.to_dict("records")]
    seeds = [s for s in seeds if s.get("ISBN")]
    por_isbn = {so_digitos(s["ISBN"]): s for s in seeds}
    log.info(f"[ponte] {len(por_isbn)} edicoes PT-BR no seed")

    cache: Dict[str, dict] = ler_json(C.CACHE_PONTE, {})
    obras_cache: Dict[int, dict] = {int(k): x for k, x in ler_json(C.CACHE_OBRAS, {}).items()}
    if retentar_falhas:
        for i in [i for i, e in cache.items() if e.get("res") is None]:
            del cache[i]

    def salvar_caches():
        gravar_json(C.CACHE_PONTE, cache)
        gravar_json(C.CACHE_OBRAS, {str(k): x for k, x in obras_cache.items()})

    # 1) ISBN (13 e 10) apenas para os ISBNs nunca vistos
    novos = [i for i in por_isbn if i not in cache]
    if novos:
        res_isbn, inline = ponte_por_isbn(novos)
        obras_cache.update(inline)
        for i in novos:
            cache[i] = {"res": res_isbn.get(i), "emb": None}
        salvar_caches()
    log.info(f"[ponte] ISBNs novos consultados: {len(novos)} | em cache: {len(por_isbn) - len(novos)}")

    # 2) fallback (catalogo da autora + embedding; depois busca) para quem ainda nao tem resolucao
    cand = [i for i in por_isbn if cache[i]["res"] is None
            and (cache[i]["emb"] is None or (usar_emb and cache[i]["emb"] is False))]
    embedder = None
    if cand and usar_emb:
        try:
            embedder = Embedder(C.MODELO_EMB)
            log.info(f"[ponte] embeddings ativos ({C.MODELO_EMB})")
        except Exception as e:
            log.warning(f"[ponte] sem embeddings ({type(e).__name__}: {e}); catalogo usara so fuzzy")
    emb_usado = embedder is not None
    if cand:
        log.info(f"[ponte] fallback para {len(cand)} edicoes")
    for k, isbn in enumerate(cand):
        s = por_isbn[isbn]
        r = None
        if s.get("Titulo"):
            r = casar_catalogo(s["Titulo"], s.get("Autor") or "", embedder) \
                or ponte_por_busca(s["Titulo"], s.get("Autor") or "")
        cache[isbn] = {"res": r, "emb": emb_usado}
        if (k + 1) % 20 == 0:
            salvar_caches()
            log.info(f"[ponte] fallback... {k + 1}/{len(cand)}")
    if cand:
        salvar_caches()

    # 3) obras que ainda faltam no cache
    ids = {cache[i]["res"]["book_id"] for i in por_isbn if cache[i]["res"]}
    faltam = [b for b in ids if b not in obras_cache]
    if faltam:
        obras_cache.update(buscar_obras(faltam))
        salvar_caches()

    # 4) montagem das tabelas
    tipos_ed = carregar_tipos_editora()
    edicoes, nao_res, obras = [], [], {}
    for isbn, s in por_isbn.items():
        r = cache[isbn]["res"]
        if not r or r["book_id"] not in obras_cache:
            nao_res.append({k: (" | ".join(sorted(x)) if isinstance(x, set) else x) for k, x in s.items()})
            continue
        ed = r.get("ed") or {}
        bid = r["book_id"]
        if bid not in obras:
            obras[bid] = montar_obra(bid, obras_cache[bid], min_count_spice)
        via_isbn = r["metodo"] in ("isbn13_edicao", "isbn10_edicao")
        revisar = (not via_isbn) and (C.REVISAR_TODO_FALLBACK or r["score"] < C.REVISAR_ABAIXO)
        # checagem independente do metodo: a autora da edicao PT aparece entre os autores da obra no Hardcover?
        autores_hc = [x.strip() for x in obras[bid].get("Autores_Hardcover", "").split("|") if x.strip()]
        autores_pt = [x.strip() for x in (s.get("Autor") or "").split(",") if x.strip()]
        conf = any(autor_confere(ap, autores_hc) for ap in autores_pt) if autores_hc and autores_pt else None
        revisar = revisar or conf is False
        edicoes.append({
            "ISBN_PT": isbn,
            "Hardcover_Book_ID": bid,
            "GB_ID": s.get("GB_ID"),
            "Titulo_PT": s.get("Titulo"),
            "Titulo_PT_Limpo": limpar_titulo(s.get("Titulo"), obras[bid].get("Titulo_Original_EN")),
            "Autor_PT": s.get("Autor"),
            "Editora_PT": s.get("Editora"),
            "Tipo_Editorial_Editora": tipo_editorial(s.get("Editora"), tipos_ed),
            "Ano_Publicacao_PT": s.get("Ano"),
            "Num_Paginas_PT": s.get("Paginas_GB") or ed.get("pages"),
            "Capa_URL_PT": s.get("Capa_URL"),
            "Capa_URL_PT_HR": s.get("Capa_URL_HR"),
            "Capa_URL_Edicao_HC": (ed.get("image") or {}).get("url") if ed else None,
            "Sinopse_PT": s.get("Sinopse"),
            "Categorias_GB": s.get("Categorias_GB"),
            "Maturity_GB": s.get("Maturity_GB"),
            "E_Ebook": s.get("E_Ebook"),
            "Metodo_Ponte": r["metodo"],
            "Score_Ponte": r["score"],
            "Score_Fuzzy": r.get("fuzzy"),
            "Score_Emb": r.get("emb"),
            "Autor_Confere": conf,
            "Revisar_Ponte": revisar,
            "Consultas_Origem": s.get("Consultas_Origem"),
            "Tipos_Origem": s.get("Tipos_Origem"),
            "Estratos_Origem": s.get("Estratos_Origem"),
            "Origens_Pais": s.get("Origens_Pais"),
        })

    df_ed = pd.DataFrame(edicoes)
    df_ob = pd.DataFrame(list(obras.values()))
    if not df_ed.empty:
        df_ob["N_Edicoes_PT_no_Seed"] = df_ob["Hardcover_Book_ID"].map(
            df_ed["Hardcover_Book_ID"].value_counts()).fillna(0).astype(int)
        df_ed.to_csv(C.OUT_EDICOES, index=False, encoding="utf-8")
        df_ob.to_csv(C.OUT_OBRAS, index=False, encoding="utf-8")
        livros = df_ed.merge(df_ob, on="Hardcover_Book_ID", how="left")
        livros.to_csv(C.OUT_LIVROS, index=False, encoding="utf-8")
        aud = livros[livros["Revisar_Ponte"]][[
            "ISBN_PT", "Titulo_PT", "Autor_PT", "Hardcover_Book_ID", "Titulo_Original_EN", "Autor_Original_EN",
            "Autores_Hardcover", "Autor_Confere", "Metodo_Ponte", "Score_Ponte", "Score_Fuzzy", "Score_Emb"]
        ].sort_values(["Autor_Confere", "Score_Ponte"])
        aud["Confere_Manual"] = ""       # preencher S/N na auditoria; vira precisao por metodo
        aud.to_csv(C.AUDITORIA_CSV, index=False, encoding="utf-8")
    pd.DataFrame(nao_res).to_csv(C.NAO_RES_CSV, index=False, encoding="utf-8")
    relatorio(df_ed, df_ob, len(por_isbn), len(nao_res))
    return df_ed, df_ob


def relatorio(df_ed: pd.DataFrame, df_ob: pd.DataFrame, n_seed: int, n_nao: int):
    p = lambda a, b: f"{(100 * a / b):.1f}%" if b else "n/a"
    print("\n" + "=" * 76)
    print("RELATORIO v3")
    print("=" * 76)
    print(f"Edicoes PT-BR no seed:        {n_seed}")
    print(f"Resolvidas (edicao->obra):    {len(df_ed)} ({p(len(df_ed), n_seed)})")
    print(f"Nao resolvidas:               {n_nao}  -> {C.NAO_RES_CSV}")
    if df_ed.empty:
        print("Nenhuma obra resolvida.")
        return
    print("Por metodo da ponte:          ", df_ed["Metodo_Ponte"].value_counts().to_dict())
    print("Ponte por e-book x impresso:  ")
    print(pd.crosstab(df_ed["E_Ebook"].fillna("NA"), df_ed["Metodo_Ponte"]).to_string())
    print(f"Revisar_Ponte (-> auditoria): {int(df_ed['Revisar_Ponte'].sum())}  -> {C.AUDITORIA_CSV}")
    n = len(df_ob)
    print(f"\nObras unicas:                 {n}  (edicoes/obra media: {len(df_ed) / n:.2f})")
    print(f"Autor_Confere: {df_ed['Autor_Confere'].value_counts(dropna=False).to_dict()}  (False = autor PT nao aparece na obra)")
    for c in ["Flag_Romance", "Flag_New_Adult", "Flag_Formato_Excluir", "Flag_Spice_Tag", "Flag_CW_Sexual_Content",
              "Flag_Closed_Door_Tag"]:
        print(f"  {c:<24} {int(df_ob[c].sum())} ({p(df_ob[c].sum(), n)})")
    print("Cobertura dos sinais de rotulacao (viabilidade da triangulacao):")
    for c in ["Generos_Hardcover", "Moods_Hardcover", "Tags_Gerais_Hardcover", "Content_Warnings", "Spice_Tags"]:
        k = int((df_ob[c].fillna("") != "").sum())
        print(f"  {c:<24} >=1 item: {k} ({p(k, n)})")
    ne = len(df_ed)
    print(f"Edicoes com capa: {int(df_ed.Capa_URL_PT.notna().sum())}/{ne} | sinopse: "
          f"{int(df_ed.Sinopse_PT.notna().sum())}/{ne} | paginas: {int(df_ed.Num_Paginas_PT.notna().sum())}/{ne}")
    print("Edicoes por tipo editorial:   ", df_ed["Tipo_Editorial_Editora"].value_counts(dropna=False).to_dict())
    print("Edicoes por estrato de origem:", df_ed["Estratos_Origem"].fillna("(sem)").str.split(" \\| ").explode()
          .value_counts().to_dict())
    print(f"Saidas: {C.OUT_EDICOES}\n        {C.OUT_OBRAS}\n        {C.OUT_LIVROS}")
    print("=" * 76)