# Estrutura do Pôster — Mismatch Analysis (POC I)

> Documento de trabalho para montar o pôster de apresentação. Todos os números abaixo foram extraídos de `outputs/tables/*.csv` e `outputs/reports/*.txt` (execução de 2026-11-01). Figuras referenciadas estão em `outputs/figures/` (PDF vetorial + PNG 300 dpi).

---

## 0. Decisões de alto nível (leia antes de diagramar)

| Item | Recomendação |
|---|---|
| **Formato** | A0 retrato (84,1 × 118,9 cm) ou A1 retrato. Layout em **3 colunas**. |
| **Mensagem única** | *"Para estudar se a capa engana, primeiro precisávamos saber o que o livro contém — e descobrimos que a sinopse brasileira não conta."* |
| **Fio condutor** | Problema → Como construímos a base → 3 achados → Limites → Próximo passo |
| **Número de figuras** | 5 (uma por bloco de resultado + diagrama do pipeline). Não mais que isso. |
| **Paleta** | Já usada nas figuras: **Okabe-Ito** (azul `#0072B2`, laranja `#E69F00`). Use as mesmas cores no pôster para coerência. |
| **Fonte** | Título ≥ 72 pt, cabeçalhos de seção ≥ 40 pt, corpo ≥ 24 pt, legendas ≥ 18 pt. Legível a 1,5 m. |
| **Texto total** | Máximo ~600 palavras. Tudo em *bullets* ou frases curtas. |
| **O que NÃO entra** | Tabela de 94 colunas, dicionário de dados, 8 regras completas, requisitos de instalação, estrutura do repositório. Isso vai no QR code. |

> [!IMPORTANT]
> O pôster deve ser **honesto sobre o estágio**: é a POC I (construção da base), não o estudo de capas. O achado forte é a omissão editorial; o achado metodológico é o viés de popularidade (MNAR). A validação humana está **parcial** (12/96 obras) — reportar como "em andamento", não como resultado.

---

## 1. Cabeçalho (faixa superior, largura total)

**Título (sugestões — escolha uma):**
- *Construção de uma base de dados para estudar a discrepância entre aparência e conteúdo de romances no Brasil* (título do relatório; longo, mas fiel)
- *Capa fofa, conteúdo adulto? Uma base de dados para medir a discrepância capa–conteúdo em romances brasileiros* (mais chamativo)
- *O que a sinopse não conta: ligando edições brasileiras ao que leitores dizem sobre elas*

**Subtítulo:** Engenharia de dados, resolução de entidades bibliográficas e rotulagem por triangulação de 5 fontes

**Autoria:** Gabriella de Lima Araujo — Departamento de Ciência da Computação, UFMG — `gabriella.lima@dcc.ufmg.br`

**Logos:** UFMG, DCC. (Canto superior direito ou esquerdo.)

**QR code** (canto inferior direito do cabeçalho ou rodapé): `https://github.com/hellolima/mismatch-analysis`

---

## 2. COLUNA ESQUERDA — Problema e Método

### 2.1 Motivação (bloco curto, ~80 palavras)

- A capa é uma **promessa**: cor, tipografia e ilustração sinalizam em um relance o gênero e o público.
- No BookTok, romances com **sexo explícito, drogas/álcool, violência e abuso** circulam com capas ilustradas em tons pastéis — estética historicamente **juvenil**.
- Crianças e adolescentes compram esses livros; a capa é o principal sinal que eles e responsáveis usam.
- **Lei nº 15.211/2025** (Estatuto Digital da Criança e do Adolescente) torna a questão de informação ao consumidor em ambiente digital.

> Elemento visual sugerido: 2–3 miniaturas de capas pastéis da base (`data/covers/`) lado a lado com um trecho de *content warning* do Hardcover. **Atenção a direitos de imagem** — se usar, cite a editora ou use desfoque/ilustração estilizada.

### 2.2 Pergunta de pesquisa e o problema anterior

**Pergunta final (POC II):** Capas juvenis estão sendo usadas para vender conteúdo adulto?

**Problema anterior (POC I, este pôster):** Não existia uma forma confiável e **independente da editora** de saber o que um livro brasileiro contém.

- A sinopse não serve: é parte do fenômeno.
- Ler centenas de livros é inviável.
- Solução: **folksonomia** — *tags* e *content warnings* que leitores registram em plataformas de catalogação social.

**Dois obstáculos técnicos (= o cerne do trabalho):**
1. Plataformas com essa informação são **globais e em inglês**; catalogam pela obra original. *Quarta Asa* ≠ *Fourth Wing* letra a letra → **resolução de entidades**.
2. Alertas são **escassos** e sua ausência **não é neutra**: livro pouco lido pode não ter alerta porque ninguém marcou → dado **MNAR**.

### 2.3 Contribuições (bullets, destacados em caixa)

- ✅ Pipeline **reprodutível** (5 etapas, checkpoint incremental, escrita atômica) que coleta edições PT-BR, liga à obra original e baixa capas.
- ✅ Taxonomia de **4 eixos** de conteúdo sensível com vocabulário bilíngue explícito (130 termos).
- ✅ **Rótulo triangulado** por obra combinando 5 fontes, que **admite não saber** (abstenção).
- ✅ Base com **252 obras / 262 edições / 49 editoras**, capa validada, sinopse, rótulo em 4 eixos e 2 alvos (com e sem sinopse).
- ✅ Protocolo de validação humana cega + ablação + sensibilidade.

### 2.4 Como a base foi construída (FIGURA 1 — diagrama do pipeline)

Reproduzir o diagrama TikZ de `relatorio.tex` (fig:pipeline) em tamanho grande, ou refazer como fluxo horizontal:

```
[Listas de busca]      [Hardcover: tags + content warnings]
 autoras (500+),                    │
 termos, editoras,                  ▼
 títulos BookTok  ──► [Google Books] ──► [PONTE edição↔obra] ──► [RÓTULO: 5 evidências] ──► [BASE ROTULADA]
                        edições PT-BR           │                                                ▲
                        sinopse + capa          └──► [Capas: download + validação] ─────────────┘
                                                                     [Validação humana cega] ┄┄┄┘
```

**Legenda curta por etapa (uma linha cada):**

| Etapa | O que faz | Detalhe que vale citar |
|---|---|---|
| **seed** | Google Books API, filtro ISBN `978-85`/`978-65` | 4 canais: autora, termo (gênero/tropo/conteúdo/BookTok), editora, título. Canal de origem é guardado para controle de circularidade. |
| **ponte** | Resolução de entidades Edição (PT-BR) → Obra (Hardcover) em 7 camadas | ISBN-13 → ISBN-10 → limpeza do título → catálogo da autora → **fuzzy ≥ 0,90 OU embeddings multilíngues (MiniLM-L12) ≥ 0,75 com margem ≥ 0,08 sobre o 2º** → busca textual → auditoria. Guarda estrita de autoria. |
| **rotulagem** | 5 fontes votam alto/baixo/abstêm; 8 regras ordenadas | Fontes: Spice_Nivel (tags), vocabulário nas tags, vocabulário na sinopse PT, *maturity rating*, *content warnings*. |
| **capas** | Download hierárquico ≥ 200 px, JPEG RGB, SHA-256 contra placeholder | Pronto para o módulo visual da POC II. |
| **eda** | Estatísticas por obra/edição, IC de Wilson/bootstrap, 21 figuras | Unidade de análise: **Obra** (FRBR) para conteúdo, **Edição** para capa. |

### 2.5 Os 4 eixos e o rótulo (tabela compacta)

| Eixo | Leitores marcam (EN) | Sinopse diz (PT) |
|---|---|---|
| 1. Sexo / erotismo | *sexual content, explicit, steamy, dark romance, rape* | conteúdo adulto, cenas quentes, +18, erótico |
| 2. Drogas / álcool | *alcoholism, drug use, addiction, overdose* | alcoolismo, embriaguez, dependência química |
| 3. Violência | *violence, murder, torture, kidnapping, gore* | violência física, tortura, sequestro, massacre |
| 4. Abuso / gatilhos | *abuse, toxic relationship, self-harm, suicide* | abuso psicológico, relacionamento tóxico, suicídio |

**Rótulo final por obra** (caixinhas coloridas):
- 🔴 **alto** — evidência convergente (*forte*: ≥ 2 fontes independentes ou *content warning*; *fraca*: 1 fonte)
- 🟢 **baixo** — evidência explícita de leveza (*closed door*, *clean*)
- 🟡 **provavelmente baixo** — nenhum voto, mas obra YA com boa cobertura comunitária (**inferência, não medição**)
- ⚪ **indeterminado** — abstenção honesta

**3 decisões de desenho contra-intuitivas** (destacar):
1. **Abstenção**: ausência de evidência nunca vira "baixo".
2. **Guarda de circularidade**: se o livro entrou via busca "conteúdo adulto", a sinopse sozinha não vale como evidência.
3. **Rótulo sem sinopse**: alvo alternativo para evitar *data leakage* ao treinar modelos de texto.

---

## 3. COLUNA CENTRAL — Resultados

### 3.1 Funil: do que coletamos ao que usamos (FIGURA 2 — `fig_16_funil`)

| Etapa | N | % do seed |
|---|---|---|
| Edições no seed (Google Books, ISBN BR) | **646** | 100 % |
| Resolvidas na ponte (→ obra Hardcover) | **291** | 45 % |
| Em escopo (ano ≥ 2015) | 262 | 40,6 % |
| Obras únicas (dedup FRBR) | **252** | 39 % |
| Rotuladas (alto / baixo / prov. baixo) | 124 | 19,2 % |
| — das quais *alto* | **107** | 16,6 % |

**Caixa de destaque:** A perda é **desigual**: 3 em cada 4 edições achadas por **autora** foram ligadas (152/203), contra 1 em cada 4 das achadas por **termo de conteúdo** (42/173). → Os livros mais explícitos do catálogo nacional (autoras brasileiras, autopublicação) são justamente os que menos existem na plataforma internacional.

### 3.2 ACHADO 1 — A informação depende da popularidade (FIGURA 3 — `fig_11_cobertura_popularidade`)

Heatmap: fração de obras com ≥ 1 tag, por quartil de popularidade.

| Quartil | Gêneros | Moods | Tags gerais | **Content warnings** |
|---|---|---|---|---|
| Q1 (menos lidas) | 30 % | 3 % | 3 % | **1,6 %** |
| Q4 (mais lidas) | 100 % | 100 % | 100 % | **68,3 %** |

- Correlação nº de leitores × nº de tags: **ρ = 0,91**.
- Proporção de *alto* sobe de **11 % (Q1) → 81 % (Q4)**.
- **Leitura:** "sem alerta" ≠ "sem conteúdo". É MNAR na prática, e é por isso que o rótulo admite *indeterminado*.

### 3.3 O que o rótulo diz sobre 252 obras (FIGURA 4 — `fig_15_eixos_sensiveis`, opcionalmente `fig_14_spice_label` como mini-donut)

**Distribuição do rótulo:**
- alto: **107 (42 %)** — 66 forte + 41 fraca
- provavelmente baixo: 17 (7 %)
- indeterminado: **128 (51 %)** — 104 deles na metade menos popular
- baixo: **0** ← *tags de leveza não ocorrem na amostra; leitores avisam sobre o que há, não sobre o que não há*

**Prevalência por eixo (IC Wilson 95 %):**

| Eixo | Obras | % |
|---|---|---|
| Qualquer eixo | 98/252 | **38,9 %** [33,1–45,0] |
| Sexo / erotismo | 83 | 32,9 % |
| Abuso / gatilhos | 39 | 15,5 % |
| Violência | 38 | 15,1 % |
| Drogas / álcool | 18 | 7,1 % |

- Metade das obras sensíveis acumula **≥ 2 eixos**; "sexo + violência + abuso" é a 2ª combinação mais comum (`fig_18_coocorrencia`). → "Conteúdo sensível" não é uma coisa só.

### 3.4 ACHADO 2 — A sinopse não avisa (FIGURA 5 — `fig_19_omissao_editorial`) ★ **figura principal do pôster**

Entre obras com *content warning* dos leitores, a sinopse PT-BR **não menciona o tema** em:

| Eixo | Omissão |
|---|---|
| **Qualquer** | **42/61 = 69 %** [56–79] |
| Sexo | 36/48 = 75 % |
| Violência | 30/32 = 94 % |
| Abuso | 36/38 = 95 % |
| **Drogas / álcool** | **18/18 = 100 %** |

**Frase de impacto:** *Em dois terços dos livros que os leitores marcam como sensíveis, a sinopse brasileira não diz nada — e a omissão é maior justamente nos temas menos "vendáveis".*

---

## 4. COLUNA DIREITA — Robustez, Limites e Próximos Passos

### 4.1 O rótulo é robusto? (mini-tabela de ablação, ou `fig_17_ablacao` reduzida)

| Remover… | Obras que mudam | *alto forte* |
|---|---|---|
| tags de gênero | 1 | 66 |
| vocabulário | 4 | 66 |
| sinopse | 9 | 64 |
| *maturity rating* | 22 | 62 |
| **content warnings** | **33** | **15** |
| só CW (fonte isolada) | 46 | — |

- **Content warnings** são a fonte dominante; mas **nenhuma fonte isolada reproduz o resultado** (a melhor erra em 46 obras) → argumento empírico para triangular.
- Sensibilidade: *alto* não varia com os parâmetros; *provavelmente baixo* oscila entre 8 e 19 (como esperado de uma inferência).
- Nas 48 obras com ≥ 2 fontes ativas, **100 % de concordância** — mas só porque todas disseram "alto". Não há polo negativo mensurável.

### 4.2 Validação humana (em andamento — reportar com cautela)

- Protocolo: 2 anotadores, **cegos** ao rótulo automático, 96 obras sorteadas estratificadas por popularidade, eixo a eixo.
- **Status atual: 12/96 obras anotadas.**
- Concordância inter-anotadores (Cohen's κ, n = 12): Sexo 0,66 · Violência 0,66 · Abuso 0,64 · Drogas 0,35 · Geral 0,47.
- Precisão da ponte (auditoria manual): **14/16 = 87,5 %** [64–97].

> [!WARNING]
> Com n = 12 os intervalos são enormes. No pôster, mostre apenas κ e a precisão da ponte como **"resultados preliminares"**. Não exiba precisão/recall do rótulo vs. humano (recall = 0,33 com IC [0,12–0,65] só gera ruído). Se a anotação avançar antes da apresentação, atualize via `python scripts/run.py --etapas gold` (ou o comando equivalente de `gold_standard.py`).

### 4.3 Como ler o rótulo / Limites (caixa "O que a base pode e não pode dizer")

- A base é **positive-unlabelled**: *alto* = "há evidência convergente"; *indeterminado* = "não sabemos" (inclui leves **e** sensíveis não marcados).
- **Popularidade confunde**: deve ser variável de controle em qualquer análise de capas.
- **A amostra não é o mercado**: perdemos > 50 % na ponte, de forma desigual (nacionais e nicho sub-representados).
- **O vocabulário é uma escolha**: drogas/álcool pode estar mal coberto em PT.
- **Buscamos o que queríamos achar**: sem o canal "termo de conteúdo", *alto* cai de 107 → 79. Direção se mantém; magnitude depende da amostra.

### 4.4 Próximos passos (POC II)

1. Concluir validação humana (96 obras) e reportar κ e precisão por eixo.
2. Módulo visual: extrair atributos estéticos das capas (paleta, ilustração vs. foto, tipografia) — capas já higienizadas e com hash.
3. Comparar **alto** vs. **leves confirmados por humanos** (nunca vs. indeterminados), controlando popularidade.
4. Métodos adequados a dados PU (*positive-unlabelled learning*).
5. Ampliar cobertura de autoras nacionais (fontes alternativas ao Hardcover).

### 4.5 Rodapé

- **Disponibilidade:** código, vocabulário (130 termos), tabelas e figuras → QR code para `github.com/hellolima/mismatch-analysis`.
- **Stack** (uma linha, fonte pequena): Python 3.10 · Google Books API · Hardcover GraphQL · `sentence-transformers` (paraphrase-multilingual-MiniLM-L12-v2) · RapidFuzz · pandas · matplotlib · ~4,5 k linhas · suíte de testes unitários.
- **Referências** (5–6, fonte pequena): IFLA FRBR; Christen (2012) *Data Matching*; Ratner et al. (2017) Snorkel; Bekker & Davis (2020) PU learning; Little & Rubin (2019) MNAR; Lei nº 15.211/2025.

---

## 5. Mapa visual do layout (A0 retrato, 3 colunas)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  [logo UFMG]   TÍTULO GRANDE                                   [QR]    │
│                Subtítulo · Autora · DCC/UFMG                            │
├───────────────────────┬───────────────────────┬─────────────────────────┤
│ 2.1 MOTIVAÇÃO         │ 3.1 FUNIL             │ 4.1 ROBUSTEZ (ablação)  │
│  (capas pastéis +     │  FIG 2 + caixa        │  mini-tabela            │
│   content warning)    │  "perda desigual"     │                         │
│                       │                       │                         │
│ 2.2 PERGUNTA +        │ 3.2 ACHADO 1          │ 4.2 VALIDAÇÃO HUMANA    │
│  2 OBSTÁCULOS         │  FIG 3 heatmap        │  (preliminar, n=12)     │
│                       │  ρ = 0,91             │                         │
│ 2.3 CONTRIBUIÇÕES     │                       │ 4.3 O QUE A BASE PODE   │
│  (caixa com ✅)        │ 3.3 RÓTULO + EIXOS    │  E NÃO PODE DIZER       │
│                       │  FIG 4 barras         │  (caixa de limites)     │
│ 2.4 PIPELINE          │  107 alto / 128 ind.  │                         │
│  FIG 1 diagrama       │                       │ 4.4 PRÓXIMOS PASSOS     │
│  (largura da coluna)  │ 3.4 ACHADO 2 ★        │  (POC II)               │
│                       │  FIG 5 omissão        │                         │
│ 2.5 4 EIXOS + RÓTULO  │  "69 % das sinopses   │ 4.5 Disponibilidade ·   │
│  tabela + 4 caixinhas │   não avisam"         │  Stack · Referências    │
│  + 3 decisões         │  (maior destaque)     │                         │
└───────────────────────┴───────────────────────┴─────────────────────────┘
```

**Hierarquia visual:** a FIG 5 (omissão) deve ser a maior figura. Em seguida FIG 1 (pipeline) e FIG 3 (heatmap). FIG 2 e FIG 4 podem ser menores.

---

## 6. Números-chave para memorizar (para a fala de 2 minutos)

| | |
|---|---|
| **646** edições coletadas → **291** ligadas (45 %) → **252** obras finais | |
| **107** obras *alto* (42 %) · **128** indeterminadas (51 %) · **0** baixo | |
| **39 %** das obras com ≥ 1 eixo sensível; **33 %** sexo, **15 %** violência, **15 %** abuso, **7 %** drogas | |
| **69 %** das sinopses omitem o que leitores alertam; **100 %** no eixo drogas/álcool | |
| Content warnings: **1,6 %** das obras menos lidas vs. **68 %** das mais lidas; **ρ = 0,91** | |
| Ponte: **87,5 %** de precisão na auditoria manual (14/16) | |
| 5 fontes · 4 eixos · 8 regras · 7 camadas de resolução · 130 termos | |

---

## 7. Checklist de produção

- [x] Figuras do pôster geradas em `outputs/poster/` (PDF vetorial + PNG) com fonte **1,5×** maior: `python scripts/poster_figs.py` (use `--escala 1.7` se ainda ficar pequeno; `--todas` mantém as 21 figuras).
- [x] FIG 1 (pipeline) desenhada em `outputs/poster/fig_00_pipeline.pdf` (matplotlib, mesma paleta).
- [ ] Usar os **PDF** no software do pôster (vetorial); PNG só como fallback.
- [ ] Decidir sobre uso de imagens de capas (direitos autorais).
- [ ] Atualizar seção 4.2 se a anotação humana avançar.
- [ ] Revisar: nenhuma afirmação de "livros populares são mais explícitos" sem a ressalva de MNAR.
- [ ] Contar palavras: ≤ 600 no corpo.
