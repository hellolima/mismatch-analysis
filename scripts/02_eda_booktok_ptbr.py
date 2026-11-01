"""02_eda_booktok_ptbr.py -- Orquestrador da Analise Exploratoria de Dados (EDA v3.2).

Arquitetura desacoplada:
  - `pipeline_ptbr.dataset`    : filtro temporal (ano >= 2015), formato e deduplicacao por Obra (FRBR)
  - `pipeline_ptbr.eda_stats`  : calculos estatisticos (Wilson IC, Bootstrap, Spearman, n-gramas)
  - `pipeline_ptbr.eda_plots`  : geracao de figuras cientificas (Okabe-Ito, 300 dpi, PNG/PDF)
  - `pipeline_ptbr.eda_report` : exportacao de tabelas, auditorias e relatorio textual

Uso:
  python scripts/02_eda_booktok_ptbr.py [--ano-min 2015] [--titulos] [--formatos png pdf]
"""
import os
import sys
import logging
import warnings
import argparse
from pathlib import Path
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_ptbr import config as C                               # noqa: E402
from pipeline_ptbr.dataset import deduplicar_por_obra               # noqa: E402
from pipeline_ptbr.eda_stats import (                               # noqa: E402
    carregar_e_preparar_eda,
    analisar_tags,
    analisar_distribuicoes,
    analisar_categoricas,
    analisar_spice_cobertura,
    calcular_ngramas_corpora,
)
from pipeline_ptbr.eda_report import (                              # noqa: E402
    exportar_tabelas_e_auditorias,
    gerar_relatorio_eda,
)

warnings.filterwarnings("ignore")
log = logging.getLogger("pipeline.eda")


def executar_eda(ano_min: int = C.ANO_MIN_PADRAO,
                 fig_titulos: bool = False,
                 formatos: Sequence[str] = ("png", "pdf"),
                 caminho_entrada: Path = None) -> None:
    C.garantir_dirs()

    if caminho_entrada is None:
        rotulado = Path(C.OUT_LIVROS_ROTULADO)
        caminho_entrada = rotulado if rotulado.exists() else Path(C.OUT_LIVROS)

    log.info(f"EDA: PT-BR Consolidado | POC I | filtro ano >= {ano_min}")
    df, meta_filtro, aud_ed = carregar_e_preparar_eda(caminho_entrada, ano_min)
    if df.empty:
        log.error("Sem dados apos carregamento/filtro. Abortando.")
        sys.exit(1)

    ob = deduplicar_por_obra(df)
    stats_tags = analisar_tags(ob)
    stats_dist = analisar_distribuicoes(df)
    stats_cat = analisar_categoricas(df)
    extra = analisar_spice_cobertura(df)
    tab_ngrams, para_plot_wc = calcular_ngramas_corpora(df)

    exportar_tabelas_e_auditorias(df, stats_tags, tab_ngrams, aud_ed)
    gerar_relatorio_eda(df, meta_filtro, stats_tags, stats_dist, extra, tab_ngrams)

    try:
        from pipeline_ptbr.eda_plots import (
            aplicar_estilo,
            plotar_wordclouds,
            gerar_visualizacoes,
        )
        aplicar_estilo()
        fmt_tuple = tuple(formatos)
        plotar_wordclouds(para_plot_wc, fmt_tuple, fig_titulos)
        gerar_visualizacoes(df, stats_tags, stats_dist, stats_cat, extra, fmt_tuple, fig_titulos)
    except ImportError as e:
        log.warning(f"Visualizacoes graficas ignoradas ({e}). Tabelas e eda_report.txt foram gerados com sucesso.")

    log.info("\n" + "=" * 90 + "\nPIPELINE DE EDA CONCLUIDO\n" + "=" * 90)
    log.info(f"Figuras: {C.OUT_FIGURES} | Tabelas: {C.OUT_TABLES} | Relatorios: {C.OUT_REPORTS}")


def main() -> None:
    ap = argparse.ArgumentParser(description="EDA BookTok PT-BR (v3.2)")
    ap.add_argument("--entrada", type=Path, default=None, help="CSV de entrada (padrao: livros_ptbr_v3_rotulado.csv ou livros_ptbr_v3.csv)")
    ap.add_argument("--ano-min", type=int, default=C.ANO_MIN_PADRAO, help="ano minimo da edicao PT-BR (padrao 2015)")
    ap.add_argument("--titulos", action="store_true", help="inclui titulos dentro das figuras")
    ap.add_argument("--formatos", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    args, _ = ap.parse_known_args()

    C.garantir_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(C.LOGS_DIR, "eda_execution.log"), encoding="utf-8"),
        ],
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("fontTools").setLevel(logging.WARNING)

    executar_eda(
        ano_min=args.ano_min,
        fig_titulos=args.titulos,
        formatos=args.formatos,
        caminho_entrada=args.entrada,
    )


if __name__ == "__main__":
    main()