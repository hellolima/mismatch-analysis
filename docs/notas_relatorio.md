# Notas brutas da autora (retiradas do relatorio.tex em 2026-11-01)

A capa de um livro eh o primeiro contato que um leitor tem com o obra. Com o crescimento da utilizacao das redes sociais, comunidades literarias online tem aumentado e, consequentemente, a divulgacao desses livros. No ambito do genero romance (romance romantico) ha livros que abordam temas como conteudo sexual, abuso etc e tal. Este trabalho busca invetigar o verdadeiro conteudo de um livro, para, posteriomente, investigar a coerencia entre a capa de um livro e seu verdadeiro conteudo. Para tal, foi-se utilizao a folksonomia (tags por usuario), sinopse dos livros. Alem disso, um databse em pt-br foi consolidado.

---

A capa de um livro eh o primeiro contato do leitor/pessoa qualquer com a obra. Expectativas sobre o livro sao criadas pela capa. A capa nao necessariamente transmite o verdadeiro conteudo (relacionar com thumnails enganosas e clickbaits). O conteudo explicito (no caso, o que de fato ha no livro) nao eh necessariamente mostrado na sinopse. Criancas e adolescentes (pessoas fora do publico alvo) estao expostas e podem comprar algo e ler algo que ira causar a elas algum tipo de dano e/ou exposicao exacerbada. Leis no Brasil mais "duras" contra a protecao de criancas e adolescentes no ambiente digital. aumento no uso das redes sociais, aumento na divulgacao, aumento no acesso, aumento no deseo (tiktok is helping us to selling more article).

---

Como aparentemente nao necessariamente a capa e a sinopse transmitem o real conteudo do livro, precisariamos de uma opiniao dos leitores sobre o que de fato o livro trata --> plataforma em  que leitroes colocam suas resenhas literarias. a api das mais conhecidas (skoob, goodreads) eh fechada --> solucao encontrada foi utilizar o haardcover. entretanto, no hardcover tudo eh cadastrado pelos usuarios e amiorai das coisas estavam em ingles --> auxilio do google books utilizando a api aberta que e disponibilizada para coletar sinopse e baixar imagens da capa na edicao em portugues para etapa posterior (poc 2). dificuldades encontrads 1) relacionar edicoes em pt-br com as edicoes em en (nao poderiamos simplesmente traduzir o titulo pois esse tipo de traducao nem sempre eh literal)
2) conseguir uma quantidade razoavel de livros: nem sempre o genero do google books batia com oq queriamnos etc e tal. para isso foram utilizados seeds: editoras br, autores q escrevem romances (dataset no kagle com mts livros do catalogo da amazon separados por genero --> peguei autores do genero romance), titulos famosos, termos comuns (tropes, palavras etc e tal). pipeline, e por fim eda.

importante: o ano a ser considerado eh o ano de publicaco no br que nao necessariamente eh o ano de publicao da oobra original
