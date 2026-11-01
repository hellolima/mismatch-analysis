"""
pipeline_ptbr: seed (Google Books) -> ponte (Hardcover) -> rotulagem triangulada -> capas -> EDA.

  config              caminhos, tokens, hiperparametros
  common              normalizacao textual, ISBN-10/13, limpeza de titulos e casamento de autoria
  publishers          chave canonica de editoras e deteccao de pares suspeitos
  dataset             filtro canonico de escopo (ano >= 2015, formato) e deduplicacao por Obra (FRBR)
  util                leitura de listas de seeds, sanitizacao de escalares e JSON atomico
  gbooks              etapa 1: seed por autoras/termos/editoras/titulos, com checkpoint JSONL
  hardcover           cliente GraphQL, ponte por ISBN, catalogo da autora, embeddings
  labeling            extracao de tags com contagem, sinais de spice (Spice_Nivel) e flags
  bridge              etapa 2: orquestra a ponte (incremental, com cache) e gera os CSVs
  rotulagem           etapa 3: rotulo de conteudo sensivel por triangulacao de fontes (Sensivel_Label)
  diagnostico_rotulo  diagnostico de cobertura, circularidade e recall relativo do rotulo
  covers              etapa 4: baixa, valida dimensao e versiona capas com hash SHA-256
  eda_stats           calculos estatisticos da EDA (Wilson IC, Bootstrap, Spearman, n-gramas)
  eda_plots           geracao de figuras cientificas (Okabe-Ito, 300 dpi, PNG/PDF)
  eda_report          exportacao de tabelas e relatorio consolidado da EDA
  probes              diagnosticos exploratorios do schema GraphQL do Hardcover

Ponto de entrada unico: scripts/run.py
"""
