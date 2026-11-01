# Protocolo de Anotação — Gold Standard (conteúdo sensível, 4 eixos)

## Arquivos
- `anotacao_eixos_cega.csv`: preencher. Uma linha por obra. **Não consulte** `anotacao_eixos_chave.csv` antes de terminar.
- `auditoria_ponte_amostra.csv`: preencher `Confere_Manual` com `S` (a obra do Hardcover é a mesma da edição PT-BR) ou `N`.

## Quem anota
Dois anotadores independentes (A e B). Cada um preenche **apenas** as suas colunas (`A_*` ou `B_*`), sem ver as do outro.
Sugestão prática: duplique o arquivo, cada anotador preenche a sua cópia, depois junte as colunas.

## O que ler para decidir
Use SOMENTE o que está na linha: `Sinopse_PT`, `Generos_Hardcover`, `Moods_Hardcover`, `Tags_Gerais_Hardcover`,
`Content_Warnings`, `Categorias_GB`. Pode usar conhecimento prévio da obra se tiver lido; anote isso em `*_Obs`.

## Escala (igual para cada eixo e para o Geral)
| Valor | Significado |
| :---: | :--- |
| `1` | o tema está presente de forma relevante na obra |
| `0` | há evidência de que o tema está ausente / a obra é leve |
| `?` | não dá para saber com o material disponível (abstenção) |

## Definição operacional de cada eixo
### Sexo — Sexo / Erotismo / Spice
- Marque `1` se houver cena, tema central ou alerta explícito. Exemplos de sinais (EN): sexual content, explicit sexual content, rape, sexual assault, sexual violence, sexual harassment. (PT): conteudo adulto, conteudo explicito, conteudo sexual, cenas? (?:de sexo|explicitas?|quentes|picantes?), sexo explicito.
- Menção de passagem (ex.: 'tomou uma taça de vinho') **não** conta para `1` em Drogas/Álcool; alcoolismo, embriaguez recorrente ou uso de drogas contam.
- Violência sexual (estupro, assédio) marca `1` em **Sexo** E em **Abuso_Gatilhos**.

### Violencia — Violência / Crime / Brutalidade
- Marque `1` se houver cena, tema central ou alerta explícito. Exemplos de sinais (EN): violence, graphic violence, gun violence, murder, homicide, blood. (PT): violencia (?:extrema|fisica|grafica|psicologica|domestica), agressao, tortura, sequestro, carcere privado.
- Menção de passagem (ex.: 'tomou uma taça de vinho') **não** conta para `1` em Drogas/Álcool; alcoolismo, embriaguez recorrente ou uso de drogas contam.
- Violência sexual (estupro, assédio) marca `1` em **Sexo** E em **Abuso_Gatilhos**.

### Drogas_Alcool — Drogas / Bebidas Alcoólicas / Vício
- Marque `1` se houver cena, tema central ou alerta explícito. Exemplos de sinais (EN): alcohol, alcoholism, drinking, drunk\w*, drug use, drug abuse. (PT): uso de drogas, trafico de drogas, abuso de drogas, drogas ilicitas, substancias? ilicitas?.
- Menção de passagem (ex.: 'tomou uma taça de vinho') **não** conta para `1` em Drogas/Álcool; alcoolismo, embriaguez recorrente ou uso de drogas contam.
- Violência sexual (estupro, assédio) marca `1` em **Sexo** E em **Abuso_Gatilhos**.

### Abuso_Gatilhos — Abuso / Relacionamento Tóxico / Saúde Mental
- Marque `1` se houver cena, tema central ou alerta explícito. Exemplos de sinais (EN): child abuse, domestic abuse, emotional abuse, physical abuse, abuse, self[\s-]?harm. (PT): abuso (?:fisico|emocional|psicologico|sexual|infantil|domestico), relacionamento toxico, suicidio, ideacao suicida, automutilacao.
- Menção de passagem (ex.: 'tomou uma taça de vinho') **não** conta para `1` em Drogas/Álcool; alcoolismo, embriaguez recorrente ou uso de drogas contam.
- Violência sexual (estupro, assédio) marca `1` em **Sexo** E em **Abuso_Gatilhos**.

### Geral — a obra tem conteúdo sensível/problemático (qualquer eixo) a ponto de ser inadequada para leitores < 16 anos?
- `1` se qualquer eixo for `1` com peso relevante; `0` se a obra for claramente leve (YA/romcom clean); `?` caso contrário.

## Decisões de desempate (anote em Obs)
- `dark romance` na sinopse/tag ⇒ pelo menos Sexo=`1` ou `?`, nunca `0`.
- `Young Adult` sozinho **não** implica `0`: YA pode ter violência/abuso. Decida pelo conteúdo.
- Se a sinopse for só marketing (sem enredo), prefira `?`.

## Depois de preencher
```bash
python scripts/pipeline_ptbr/gold_standard.py avaliar
```
Gera κ entre anotadores, precisão/recall do rótulo automático por eixo e precisão da ponte por método.