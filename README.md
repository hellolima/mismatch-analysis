# BookTok Mismatch Analysis (PT-BR)

Pipeline de Engenharia de Dados, Resolução de Entidades Bibliográficas (*Record Linkage*) e Rotulagem por Triangulação para análise de discrepância visual-textual (*mismatch*) em edições brasileiras do ecossistema BookTok.

> **Documentação Arquitetural Completa:** Consulte [`ARQUITETURA_E_PIPELINE.md`](./ARQUITETURA_E_PIPELINE.md) para detalhes matemáticos da ponte de resolução em 7 camadas, modelo conceitual FRBR, prevenção de *data leakage* e dicionário completo de variáveis.

---

## Visão Geral do Problema

Livros comercializados no mercado editorial brasileiro com alto teor de **conteúdos sensíveis e "problemáticos" no geral** — englobando **sexo/erotismo (*spice*), uso/abuso de drogas, consumo de bebidas alcoólicas/alcoolismo, violência explícita/crime e gatilhos de abuso/saúde mental** — frequentemente adotam capas ilustradas em tons pastéis associadas à estética infantojuvenil (*Young Adult*). Este repositório implementa a infraestrutura reprodutível da **POC I**:

1. **Ingestão Amostral Controlada (`seed`):** Coleta de edições brasileiras (ISBN `978-85` / `978-65`) via Google Books API a partir de autoras estratificadas, termos/tropos, editoras e títulos-âncora.
2. **Resolução de Entidades (`ponte`):** Ligação determinística (ISBN-13/ISBN-10) e probabilística/semântica (Catálogo da Autora + *Sentence Embeddings* Multilíngues `MiniLM-L12` + Fuzzy Matching) entre a **Edição PT-BR** e a **Obra Global** na plataforma **Hardcover**.
3. **Rotulagem Triangulada Multi-Eixo (`rotulagem`):** Classificação de conteúdo sensível (`Sensivel_Label`) e decomposição em **4 eixos temáticos** (`Flag_Eixo_Sexo`, `Flag_Eixo_Drogas_Alcool`, `Flag_Eixo_Violencia`, `Flag_Eixo_Abuso_Gatilhos`), com abstenção explícita para dados faltantes não-aleatórios (MNAR), combinando folksonomia, *Content Warnings*, sinopse PT-BR e *maturity rating*.
4. **Ingestão Visual (`capas`):** Download hierárquico de capas em alta resolução ($\ge 200\text{ px}$), padronização RGB JPEG e expurgo criptográfico (SHA-256) de *placeholders*.
5. **Análise Exploratória (`eda`):** Estatísticas descritivas no nível da Obra e da Edição, intervalos de confiança de Wilson/Bootstrap para cada eixo sensível e figuras vetorizadas (PDF/PNG) na paleta Okabe-Ito.

---

## Instalação e Configuração

### 1. Pré-requisitos
* **Python 3.10+**
* Chaves de API para **Hardcover** (obrigatória para a etapa `ponte`) e **Google Books** (opcional, recomendada para cotas maiores na etapa `seed`).

### 2. Ambiente Virtual e Dependências
```bash
python -m venv .venv
source .venv/bin/activate

# Dependências principais (coleta, ponte léxica, capas e EDA)
pip install -r requirements.txt

# Opcional (Recomendado): Embeddings multilíngues para tradução de títulos na ponte
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-embeddings.txt
```

### 3. Variáveis de Ambiente
Copie o arquivo de exemplo e preencha seus tokens:
```bash
cp .env.example .env
```
Edite `.env`:
```ini
HARDCOVER_API_TOKEN=seu_token_aqui
GOOGLE_BOOKS_API_KEY=sua_chave_aqui
```

---

## Como Executar o Pipeline

O ponto de entrada unificado é [`scripts/run.py`](./scripts/run.py). Todas as etapas possuem **checkpoint incremental e escrita atômica** (podem ser interrompidas com `Ctrl+C` e retomadas sem perda de progresso ou gasto duplicado de cota de API).

```bash
# 1. Execução de teste rápido (amostra pequena: 5 autoras, 3 termos, 30 edições)
python scripts/run.py --teste

# 2. Execução completa (seed -> ponte -> rotulagem -> capas -> eda)
python scripts/run.py

# 3. Executar apenas etapas específicas
python scripts/run.py --etapas ponte,rotulagem,eda

# 4. Executar diagnóstico de cobertura, circularidade e recall do rótulo triangulado
python scripts/run.py --sondar rotulo

# 5. Executar apenas a EDA com figuras em PNG e PDF (filtro temporal >= 2015)
python scripts/02_eda_booktok_ptbr.py --ano-min 2015 --formatos png pdf

# 6. Rodar a suíte de testes unitários
python -m unittest discover -s tests -v
```

---

## Estrutura do Repositório

```text
mismatch-analysis/
├── README.md                          # Guia rápido de instalação, execução e arquitetura
├── ARQUITETURA_E_PIPELINE.md          # Documentação científica e especificação técnica completa
├── requirements.txt                   # Dependências principais (Core + EDA + Capas)
├── requirements-embeddings.txt        # Dependência opcional (Sentence Transformers)
├── .env.example                       # Modelo de variáveis de ambiente
├── scripts/
│   ├── run.py                         # Orquestrador CLI unificado (seed -> ponte -> rotulagem -> capas -> eda)
│   ├── 02_eda_booktok_ptbr.py         # Orquestrador da Análise Exploratória de Dados (EDA)
│   ├── ptbr_common.py                 # Re-export de compatibilidade para pipeline_ptbr.common
│   ├── editoras_norm.py               # Re-export de compatibilidade para pipeline_ptbr.publishers
│   └── pipeline_ptbr/
│       ├── config.py                  # Caminhos centralizados, tokens e hiperparâmetros
│       ├── common.py                  # Normalização textual, aritmética ISBN-10/13, guarda de autoria e limpeza de títulos
│       ├── publishers.py              # Chave canônica de editoras brasileiras e detecção de pares suspeitos
│       ├── dataset.py                 # Filtro canônico de escopo (ano >= 2015, formato) e deduplicação por Obra (FRBR)
│       ├── util.py                    # Sanitização de escalares, leitura de seeds e escrita JSON atômica
│       ├── gbooks.py                  # Etapa 1: Coleta Google Books API com checkpoint JSONL
│       ├── hardcover.py               # Cliente GraphQL Hardcover e estratégias de resolução (ISBN/Catálogo/Busca)
│       ├── bridge.py                  # Etapa 2: Orquestração incremental da ponte e backfill de proveniência
│       ├── labeling.py                # Extração da folksonomia do Hardcover e cálculo ordinal de Spice_Nivel (0-3)
│       ├── rotulagem.py               # Etapa 3: Rótulo triangulado de conteúdo sensível (Sensivel_Label v3.1)
│       ├── diagnostico_rotulo.py      # Auditoria de circularidade, cobertura e recall relativo das fontes de rótulo
│       ├── covers.py                  # Etapa 4: Download hierárquico de capas, validação Pillow e hash SHA-256
│       ├── eda_stats.py               # Cálculos estatísticos puros da EDA (Wilson IC, Bootstrap, Spearman, n-gramas)
│       ├── eda_plots.py               # Geração de figuras científicas (Okabe-Ito, 300 dpi, PNG/PDF)
│       ├── eda_report.py              # Exportação de tabelas analíticas, auditorias e relatório consolidado
│       └── probes.py                  # Diagnóstico de introspecção do schema GraphQL do Hardcover
├── tests/
│   └── test_pipeline.py               # Testes unitários das regras de negócio, ISBN, filtros e rotulagem
├── data/
│   ├── seeds/                         # Quadro amostral controlado (autores_v2.txt, termos_v2.txt, editoras.txt, titulos.txt)
│   ├── gold_standard/                 # Planilhas de anotação humana cega (anotacao_gold_cega.csv, anotacao_gold_chave.csv)
│   ├── cache/                         # Caches atômicos da ponte (gerados em execução)
│   ├── covers/                        # Capas higienizadas em JPEG RGB (geradas em execução)
│   └── processed/                     # Tabelas canônicas geradas pelo pipeline
└── outputs/
    ├── figures/                       # Gráficos científicos em PNG (300 dpi) e PDF vetorial
    ├── tables/                        # Tabelas de frequências de tags, n-gramas, cobertura e prevalências (Wilson IC)
    ├── audits/                        # Tabelas de auditoria (limpeza de títulos, editoras, conflitos de rotulagem)
    └── reports/                       # Relatórios analíticos (eda_report.txt, rotulagem_resumo.txt, diagnostico_rotulo.txt)
```
