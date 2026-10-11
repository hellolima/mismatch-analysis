"""
pipeline_ptbr: seed (Google Books) -> ponte (Hardcover) -> rotulos/tags -> capas.

  config     caminhos, tokens, hiperparametros
  util       leitura de listas de seeds, JSON atomico
  gbooks     etapa 1: seed por autoras/termos/editoras/titulos, com checkpoint
  hardcover  cliente GraphQL, ponte por ISBN, catalogo da autora, embeddings
  labeling   tags com contagem, sinais de spice, flags
  bridge     etapa 2: orquestra a ponte (incremental, com cache) e gera os CSVs
  covers     etapa 3: baixa e versiona capas
  probes     diagnosticos: schema do Hardcover e viabilidade inversa

Ponto de entrada unico: scripts/run.py
"""
