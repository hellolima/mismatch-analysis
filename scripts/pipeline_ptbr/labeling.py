"""Obra (Hardcover) -> tags com contagem, sinais de spice e flags."""
import re
import json
from typing import Dict, List, Optional, Tuple

from ptbr_common import norm


def tags_por_categoria(cached) -> Dict[str, List[Tuple[str, Optional[int]]]]:
    if isinstance(cached, str):
        try:
            cached = json.loads(cached)
        except ValueError:
            cached = None
    out: Dict[str, list] = {}
    if not isinstance(cached, dict):
        return out
    for cat, itens in cached.items():
        if not isinstance(itens, list):
            continue
        for it in itens:
            if isinstance(it, dict) and it.get("tag"):
                out.setdefault(cat, []).append((str(it["tag"]).strip(), it.get("count")))
            elif isinstance(it, str) and it.strip():
                out.setdefault(cat, []).append((it.strip(), None))
    return out


RE_FORMATO_EXCLUIR = re.compile(
    r"\b(manga|manhwa|manhua|comics?|graphic novel|light novel|webtoon|"
    r"picture book|children|kids|doujinshi)\b")
# Sinais de spice na folksonomia (o centro do projeto: o relatorio mede cobertura deles)
RE_SPICE = re.compile(r"\b(steamy|spicy|explicit|open door|erotic\w*|smut\w*|sex scenes?|"
                      r"graphic sex|high heat)\b")
# "Sexual content" e uma CONTENT WARNING (presenca de conteudo sexual), nao um nivel de spice: fica em flag propria.
RE_CW_SEXUAL = re.compile(r"\bsexual content\b")
CAT_CW = "Content Warning"
# papeis de contribuicao que NAO sao autoria (o tradutor nao e o autor da obra)
NAO_AUTOR = ("translat", "tradut", "illustrat", "ilustr", "editor", "narrat", "foreword", "introduction",
             "afterword", "cover", "adapt", "compil", "photograph")


def autores_da_obra(contribs) -> List[str]:
    """Nomes de quem escreveu a obra. Sem o campo `contribution` (schema antigo) devolve todos, como antes."""
    todos, autores = [], []
    for c in contribs or []:
        nome = ((c or {}).get("author") or {}).get("name")
        if not nome:
            continue
        todos.append(nome)
        papel = norm((c or {}).get("contribution") or "")
        if not any(k in papel for k in NAO_AUTOR) and nome not in autores:
            autores.append(nome)
    return autores or todos


RE_CLOSED = re.compile(r"\b(closed door|fade to black|clean romance|sweet romance|no spice|low heat|no sex)\b")


def sinais_regex(cats: dict, regex: "re.Pattern", min_count: int, so_cw: bool = False) -> Tuple[List[str], int, bool]:
    """so_cw=True varre apenas Content Warnings; False varre todas as categorias EXCETO Content Warnings."""
    achados, total, passa = [], 0, False
    for cat, itens in cats.items():
        if (cat == CAT_CW) != so_cw:
            continue
        for nome, cnt in itens:
            if regex.search(norm(nome)):
                c = cnt if isinstance(cnt, (int, float)) else None
                achados.append(f"{nome}:{int(c) if c is not None else '?'}")
                total += int(c) if c is not None else 1
                if c is None or c >= min_count:
                    passa = True
    return achados, total, passa


def montar_obra(book_id: int, obra: dict, min_count_spice: int) -> dict:
    cats = tags_por_categoria(obra.get("cached_tags"))
    nomes = lambda c: [t for t, _ in cats.get(c, [])]
    generos, moods, tags, cws = nomes("Genre"), nomes("Mood"), nomes("Tag"), nomes("Content Warning")
    todos = " | ".join(generos + tags).lower()
    sp, sp_total, sp_flag = sinais_regex(cats, RE_SPICE, min_count_spice)
    cl, cl_total, cl_flag = sinais_regex(cats, RE_CLOSED, min_count_spice)
    cw_sex, cw_sex_total, cw_sex_flag = sinais_regex(cats, RE_CW_SEXUAL, 1, so_cw=True)
    autores = autores_da_obra(obra.get("contributions"))
    autor_en = autores[0] if autores else None
    img = obra.get("image") or {}
    sep = " | "      # separador sem virgula: tags podem conter virgula (o EDA divide por '|')
    return {
        "Hardcover_Book_ID": book_id,
        "Titulo_Original_EN": obra.get("title"),
        "Autor_Original_EN": autor_en,
        "Autores_Hardcover": sep.join(autores),
        "Ano_Original_EN": obra.get("release_year"),
        "Avaliacao_Hardcover": obra.get("rating"),
        "Qtd_Avaliacoes_Hardcover": obra.get("ratings_count"),
        "Qtd_Usuarios_Hardcover": obra.get("users_count"),
        "Capa_URL_Original_EN": img.get("url"),
        "Generos_Hardcover": sep.join(generos),
        "Moods_Hardcover": sep.join(moods),
        "Tags_Gerais_Hardcover": sep.join(tags),
        "Content_Warnings": sep.join(cws),
        "Tags_JSON": json.dumps(obra.get("cached_tags"), ensure_ascii=False),   # mantem 'count' (peso)
        "Spice_Tags": sep.join(sp),
        "Spice_Tags_Contagem": sp_total,
        "Flag_Spice_Tag": sp_flag,
        "Closed_Door_Tags": sep.join(cl),
        "Flag_Closed_Door_Tag": cl_flag,
        "Flag_CW_Sexual_Content": cw_sex_flag,
        "CW_Sexual_Content_Contagem": cw_sex_total,
        "Flag_Romance": "romance" in todos,
        "Flag_New_Adult": ("new adult" in todos) or ("new-adult" in todos),
        "Flag_Formato_Excluir": bool(RE_FORMATO_EXCLUIR.search(todos)),
    }