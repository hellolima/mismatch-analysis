"""Camada de exportacao de tabelas analiticas, auditorias e relatorio textual da EDA."""
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from . import config as C
from .publishers import pares_suspeitos
from .dataset import deduplicar_por_obra, resolver_colunas_rotulo
from .eda_stats import (
    wilson_ic,
    prevalencias_flags,
    cobertura_por_popularidade,
    sensibilidade_min_count,
)

log = logging.getLogger("pipeline.eda.report")


def exportar_tabelas_e_auditorias(df: pd.DataFrame,
                                  stats_tags: Dict,
                                  tab_ngrams: Dict[str, pd.DataFrame],
                                  aud_ed: Optional[pd.DataFrame],
                                  tables_dir: Path = Path(C.OUT_TABLES),
                                  audits_dir: Path = Path(C.OUT_AUDITS)) -> None:
    """Exporta frequencias de tags, n-gramas e auditorias de limpeza de titulos e editoras."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    audits_dir.mkdir(parents=True, exist_ok=True)

    # 1. N-gramas (titulos e sinopses)
    for nome, tab in tab_ngrams.items():
        tab.to_csv(tables_dir / f"ngrams_{nome}.csv", index=False, encoding="utf-8")

    # 2. Frequencia de tags
    rows = []
    for cat, s in stats_tags.items():
        for tag, count in s["frequencias"].most_common():
            rows.append({
                "Tipo_Tag": cat.replace("_", " "),
                "Tag": tag,
                "Frequencia": count,
                "Percentual_%": (count / s["n_total"]) * 100 if s["n_total"] else 0,
            })
    pd.DataFrame(rows).sort_values("Frequencia", ascending=False).to_csv(
        tables_dir / "tags_frequency.csv", index=False, encoding="utf-8")
    log.info(f"  Salvo: {tables_dir / 'tags_frequency.csv'} ({len(rows)} tags)")

    # 3. Auditoria de normalizacao de editoras
    if aud_ed is not None:
        aud_ed.to_csv(audits_dir / "editoras_normalizacao_auditoria.csv", encoding="utf-8")
    if "Editora_Chave" in df.columns:
        sus = pares_suspeitos(df["Editora_Chave"].dropna().unique())
        pd.DataFrame(sus, columns=["Chave_Curta", "Chave_Longa"]).to_csv(
            audits_dir / "editoras_pares_suspeitos.csv", index=False, encoding="utf-8")

    # 4. Auditoria de limpeza de titulos
    if "Titulo_Limpo" in df.columns and "Titulo_Original_Bruto" in df.columns:
        alt = df[df["Titulo_Original_Bruto"] != df["Titulo_Limpo"]][["Titulo_Original_Bruto", "Titulo_Limpo"]]
        alt.to_csv(audits_dir / "titulos_limpeza_auditoria.csv", index=False, encoding="utf-8")
        log.info(f"  Titulos alterados pela limpeza: {len(alt)}/{len(df)}")


def gerar_relatorio_eda(df: pd.DataFrame,
                        meta_filtro: Dict[str, int],
                        stats_tags: Dict,
                        stats_dist: Dict,
                        extra: Dict,
                        tab_ngrams: Dict[str, pd.DataFrame],
                        reports_dir: Path = Path(C.OUT_REPORTS),
                        tables_dir: Path = Path(C.OUT_TABLES)) -> None:
    """Gera o relatorio textual completo (eda_report.txt) e exporta tabelas de prevalencia e cobertura."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    caminho_rel = reports_dir / "eda_report.txt"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ob = deduplicar_por_obra(df)

    with open(caminho_rel, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\nANÁLISE EXPLORATÓRIA DE DADOS (EDA)\n")
        f.write("Dataset: PT-BR Consolidado (v3.2)\n")
        f.write(f"Gerado em: {ts}\n" + "=" * 90 + "\n\n")

        m = meta_filtro
        f.write("FILTRO TEMPORAL E DE ESCOPO\n" + "-" * 90 + "\n")
        f.write(f"Critério: ano da edição PT-BR >= {m.get('ano_min')}\n")
        f.write(f"Edições no CSV: {m.get('n_inicial')} | removidas por ano < {m.get('ano_min')}: {m.get('n_antes_ano_min')} | "
                f"removidas por ano ausente/inparseável: {m.get('n_sem_ano')} | "
                f"removidas por formato fora de escopo (mangá/HQ/infantil): {m.get('n_formato_excluido', 0)} | "
                f"mantidas: {m.get('n_final')}\n\n")

        f.write("VISÃO GERAL DO DATASET (após filtro)\n" + "-" * 90 + "\n")
        f.write(f"Total de registros (edições): {len(df)}\nTotal de obras únicas: {len(ob)}\nTotal de colunas: {len(df.columns)}\n")
        if "ano" in stats_dist:
            f.write(f"Período temporal (ano da edição PT-BR): {stats_dist['ano']['min']}-{stats_dist['ano']['max']}\n")
        if "Editora_PT" in df.columns:
            f.write(f"Cobertura de Editora: {df['Editora_PT'].notna().mean() * 100:.1f}%\n")
        if "ISBN_PT" in df.columns:
            f.write(f"Cobertura de ISBN: {df['ISBN_PT'].notna().mean() * 100:.1f}%\n")
        f.write("\n")

        for chave, titulo in (("rating", "RATINGS (por obra)"), ("qtd_avaliacoes", "QTD DE AVALIAÇÕES (por obra)"),
                              ("paginas", "PÁGINAS (por edição)")):
            if chave in stats_dist:
                r = stats_dist[chave]
                f.write(titulo + "\n" + "-" * 90 + "\n")
                f.write(f"  Média: {r['mean']:.3f} | Mediana: {r['median']:.3f} | DP: {r['std']:.3f} | "
                        f"Q1-Q3: {r['q1']:.2f}-{r['q3']:.2f} | Mín-Máx: {r['min']:.2f}-{r['max']:.2f} | N: {r['count']}\n")
                if "ic_boot" in r:
                    f.write(f"  IC bootstrap 95% da média: [{r['ic_boot'][0]:.3f}, {r['ic_boot'][1]:.3f}]\n")
                f.write("\n")

        f.write("ANÁLISE DE TAGS E METADADOS\n" + "-" * 90 + "\n")
        for cat in ("Generos", "Moods", "Tags_Gerais", "Tags_Estruturadas", "Content_Warnings"):
            if cat not in stats_tags:
                continue
            s = stats_tags[cat]
            f.write(f"\n  {cat.upper()}\n    Tags únicas: {s['n_unicas']}\n    Total de tags: {s['n_total']}\n")
            f.write(f"    Registros com tag: {s['n_registros']}/{s['n_base']} obras ({s['cobertura_%']:.1f}%)\n    Top 10:\n")
            for i, (tag, fr) in enumerate(s["frequencias"].most_common(10), 1):
                pct_val = fr / s["n_total"] * 100 if s["n_total"] else 0
                f.write(f"      {i:2d}. {tag:<40} {fr:3d} ({pct_val:5.1f}%)\n")
        f.write("\n" + "=" * 90 + "\n")

        # Secao v3
        f.write("\nSEÇÃO v3: UNIDADES, PONTE E SINAIS DE CONTEÚDO SENSÍVEL (SEXO, DROGAS/ÁLCOOL, VIOLÊNCIA, ABUSO)\n" + "-" * 90 + "\n")
        f.write(f"Edições (ISBN): {len(df)} | Obras únicas: {len(ob)}\n")
        for col in ("Metodo_Ponte", "E_Ebook", "Tipo_Editorial_Editora"):
            if col in df.columns:
                f.write(f"{col}: {df[col].value_counts(dropna=False).to_dict()}\n")
        if "Revisar_Ponte" in df.columns:
            f.write(f"Revisar_Ponte (entra na auditoria Gold Standard): {int(df['Revisar_Ponte'].sum())}\n")
        for col, rot in (("Estratos_Origem", "Estratos de origem (quadro amostral proposital)"),
                         ("Origens_Pais", "Origens (país da autora no seed)"),
                         ("Tipos_Origem", "Tipos de origem (canal do seed; base da sensibilidade sem 'termo:conteudo')")):
            if col in df.columns:
                v = df[col].fillna("(sem)").str.split(r" \| ").explode().value_counts().to_dict()
                f.write(f"{rot}: {v}\n")

        if "Editora_Chave" in df.columns and "Editora_PT" in df.columns:
            f.write(f"\nEditoras: {df['Editora_PT'].dropna().nunique()} grafias brutas -> "
                    f"{df['Editora_Chave'].dropna().nunique()} chaves canônicas\n")
            if "Tipo_Editorial_Editora" in df.columns:
                desc = df[df["Tipo_Editorial_Editora"] == "desconhecido"]["Editora_Chave"].value_counts().head(15)
                if len(desc):
                    f.write(f"Editoras 'desconhecido' mais frequentes: {desc.to_dict()}\n")

        f.write("\nFlags e Eixos de Conteúdo Sensível por obra (proporção com IC de Wilson 95%):\n")
        prev = prevalencias_flags(df)
        for _, r in prev.iterrows():
            f.write(f"  {r['Flag']:<28} {int(r['k']):>4}/{int(r['n'])} = {100 * r['prop']:5.1f}%  "
                    f"[{100 * r['IC95_inf']:.1f}, {100 * r['IC95_sup']:.1f}]\n")
        prev.to_csv(tables_dir / "flags_prevalencia_wilson.csv", index=False, encoding="utf-8")

        if "Spice_Status" in ob.columns:
            f.write("\nEIXO SEXUAL / SPICE TRI-ESTADO (por obra). 'sem_sinal' = DESCONHECIDO, não é ausência de conteúdo:\n")
            f.write(f"  Spice_Status: {ob['Spice_Status'].value_counts(dropna=False).to_dict()}\n")
            k = int((ob["Spice_Status"] == "positivo").sum())
            p, lo, hi = wilson_ic(k, len(ob))
            f.write(f"  Spice positivo: {k}/{len(ob)} = {100 * p:.1f}% [IC Wilson 95%: {100 * lo:.1f}, {100 * hi:.1f}]\n")
            if "Spice_Nivel" in ob.columns:
                niv = pd.to_numeric(ob["Spice_Nivel"], errors="coerce").value_counts(dropna=False).sort_index()
                niv = {("sem_sinal" if pd.isna(a) else int(a)): int(n) for a, n in niv.items()}
                f.write(f"  Spice_Nivel (0 fechado, 1 sensual, 2 cena aberta, 3 explícito): {niv}\n")
            if "Flag_Spice_Tag" in ob.columns:
                f.write("  Flag_Spice_Tag (antigo) x Spice_Status (novo):\n    "
                        + pd.crosstab(ob["Flag_Spice_Tag"], ob["Spice_Status"]).to_string().replace("\n", "\n    ") + "\n")
            sens = sensibilidade_min_count(ob)
            if sens:
                f.write("  Sensibilidade ao limiar mínimo de votos por tag:\n")
                for kk, vv in sens.items():
                    f.write(f"    {kk}: {vv}\n")
            cob = cobertura_por_popularidade(ob)
            if cob is not None:
                cob.to_csv(tables_dir / "cobertura_por_popularidade.csv", encoding="utf-8")
                f.write("  Cobertura de tags por quartil de popularidade (queda nos quartis baixos = faltante NÃO aleatório):\n    "
                        + cob.to_string().replace("\n", "\n    ") + "\n")
            if "spearman_users_ntags" in extra:
                rho, pv, n = extra["spearman_users_ntags"]
                f.write(f"  Spearman (nº de leitores x nº total de tags): rho={rho:.3f}, p={pv:.4f}, n={n}\n")
            if "spice_por_pop" in extra:
                f.write("  Spice positivo por quartil de popularidade (Wilson 95%):\n    "
                        + extra["spice_por_pop"].round(3).to_string(index=False).replace("\n", "\n    ") + "\n")
                extra["spice_por_pop"].to_csv(tables_dir / "spice_por_popularidade.csv", index=False, encoding="utf-8")
            if "sensivel_por_pop" in extra:
                f.write("  Conteúdo Sensível Geral (alto / qualquer eixo) por quartil de popularidade (Wilson 95%):\n    "
                        + extra["sensivel_por_pop"].round(3).to_string(index=False).replace("\n", "\n    ") + "\n")
                extra["sensivel_por_pop"].to_csv(tables_dir / "sensivel_por_popularidade.csv", index=False, encoding="utf-8")

        col_lbl, col_conf, _ = resolver_colunas_rotulo(ob)
        if col_lbl in ob.columns:
            f.write(f"\nRÓTULO TRIANGULADO DE CONTEÚDO SENSÍVEL ({col_lbl}; rotulagem.py; supervisão fraca; 'indeterminado' = desconhecido)\n")
            f.write(f"  {col_lbl}: {ob[col_lbl].value_counts().to_dict()}\n")
            if col_conf in ob.columns:
                f.write("  Rótulo x confiança:\n    "
                        + pd.crosstab(ob[col_lbl], ob[col_conf]).to_string().replace("\n", "\n    ") + "\n")
                nrev = int((ob[col_conf] == "conflito").sum())
                f.write(f"  Obras em conflito entre fontes (revisão manual): {nrev}\n")

        if {"Flag_Spice_Tag", "Tipo_Editorial_Editora"} <= set(df.columns):
            f.write("\nAssociação DESCRITIVA (não causal; amostra proposital, não representa o mercado):\n"
                    "Contagem de EDIÇÕES por Flag_Spice_Tag e tipo editorial (atributo da obra replicado por edição)\n")
            tab = pd.crosstab(df["Tipo_Editorial_Editora"].fillna("NA"), df["Flag_Spice_Tag"])
            tab["n"] = tab.sum(axis=1)
            f.write(tab.to_string() + "\n")
        if {"Spice_Status", "Tipo_Editorial_Editora", "Hardcover_Book_ID"} <= set(df.columns):
            misto = int((df.groupby("Hardcover_Book_ID")["Tipo_Editorial_Editora"].nunique() > 1).sum())
            f.write(f"\nContagem de OBRAS por Spice_Status e tipo editorial (tipo da 1ª edição; {misto} obra(s) com tipos mistos):\n")
            tab2 = pd.crosstab(ob["Tipo_Editorial_Editora"].fillna("NA"), ob["Spice_Status"])
            tab2["n"] = tab2.sum(axis=1)
            f.write(tab2.to_string() + "\n")

        if tab_ngrams:
            f.write("\nTERMOS MAIS FREQUENTES (frequência de documento, nível obra; lista completa em outputs/tables/ngrams_*.csv)\n")
            for nome, t in tab_ngrams.items():
                for tipo in ("unigrama", "bigrama"):
                    top = t[t["Tipo"] == tipo].head(10)
                    f.write(f"  {nome}/{tipo}: " + ", ".join(f"{a} ({b})" for a, b in zip(top["Termo"], top["Freq_Doc"])) + "\n")
    log.info(f"  Relatorio consolidado salvo em: {caminho_rel}")
