"""Testes unitarios da infraestrutura core do pipeline BookTok PT-BR."""
import os
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from pipeline_ptbr.common import (  # noqa: E402
    autor_confere,
    eh_isbn_br,
    eh_kit_ou_box,
    indexar_autoras,
    isbn10_para_13,
    isbn13_para_10,
    limpar_html,
    limpar_titulo,
    norm,
    sem_acento,
)
from pipeline_ptbr.publishers import chave_editora, pares_suspeitos  # noqa: E402
from pipeline_ptbr.dataset import (  # noqa: E402
    deduplicar_por_obra,
    filtrar_escopo,
    resolver_colunas_rotulo,
)
from pipeline_ptbr.labeling import montar_obra, nivel_tag, spice_da_obra  # noqa: E402
from pipeline_ptbr.rotulagem import eixos_sensiveis_obra, rotular, voto_cw, voto_sinopse, voto_vocab  # noqa: E402
from pipeline_ptbr.eda_stats import wilson_ic, contar_ngramas, prevalencias_flags  # noqa: E402


class TestCommon(unittest.TestCase):
    def test_isbn_conversion(self):
        # ISBN-13 BR <-> ISBN-10 BR
        isbn13 = "9788501112514"
        self.assertTrue(eh_isbn_br(isbn13))
        isbn10 = isbn13_para_10(isbn13)
        self.assertIsNotNone(isbn10)
        self.assertEqual(isbn10_para_13(isbn10), isbn13)
        # Prefix 65 (not convertible to ISBN-10 group 85, but valid BR ISBN-13)
        self.assertTrue(eh_isbn_br("9786555601234"))
        self.assertFalse(eh_isbn_br("9780062315007"))

    def test_autor_confere_guard(self):
        # Casos criticos documentados na metodologia
        self.assertFalse(autor_confere("Christina Lauren", ["Lauren Asher"]))
        self.assertFalse(autor_confere("A.C. Meyer", ["Stephenie Meyer"]))
        self.assertTrue(autor_confere("A.C. Meyer", ["A. C. Meyer"]))
        self.assertTrue(autor_confere("Jennifer L. Armentrout", ["Jennifer Armentrout"]))
        self.assertTrue(autor_confere("Jennifer L. Armentrout", ["Jennifer L. Armentrout"]))

    def test_indexar_autoras(self):
        autoras = [
            {"nome": "Colleen Hoover", "estrato": "new_adult", "origem": "EN"},
            {"nome": "Ali Hazelwood", "estrato": "romcom", "origem": "EN"},
        ]
        idx = indexar_autoras(autoras)
        self.assertIn("hoover", idx)
        self.assertIn("hazelwood", idx)

    def test_limpar_titulo_marketing(self):
        bruto = (
            "Meu Romeu sombrio – O dark romance de L.J. Shen e Parker S. Huntington "
            "é uma releitura moderna de Romeu e Julieta"
        )
        self.assertEqual(limpar_titulo(bruto), "Meu Romeu sombrio")
        self.assertEqual(limpar_titulo("Quarta Asa (Sucesso no TikTok) - Vol. 1"), "Quarta Asa")
        self.assertEqual(limpar_titulo("A Hipótese do Amor (The Love Hypothesis)", "The Love Hypothesis"), "A Hipótese do Amor")

    def test_kit_ou_box_and_html(self):
        self.assertTrue(eh_kit_ou_box("Box Trilogia Príncipe Cruel"))
        self.assertFalse(eh_kit_ou_box("O Príncipe Cruel"))
        self.assertEqual(limpar_html("<p>Amor &amp; <b> Segredos </b><br/>Fim</p>"), "Amor & Segredos Fim")
        self.assertEqual(norm("É Assim Que Acaba!"), "e assim que acaba")
        self.assertEqual(sem_acento("É Assim: Que Acaba!"), "e assim: que acaba!")


class TestPublishers(unittest.TestCase):
    def test_chave_editora(self):
        self.assertEqual(chave_editora("Editora Arqueiro"), "arqueiro")
        self.assertEqual(chave_editora("Harlequin Brasil"), "harlequin")
        self.assertEqual(chave_editora("Grupo Editorial Record Ltda."), "record")
        self.assertIsNone(chave_editora("Editora"))
        self.assertIsNone(chave_editora(None))

    def test_pares_suspeitos(self):
        pares = pares_suspeitos(["rocco", "rocco jovens leitores", "arqueiro"])
        self.assertEqual(pares, [("rocco", "rocco jovens leitores")])


class TestDatasetAndLabeling(unittest.TestCase):
    def test_filtrar_escopo_and_deduplicar(self):
        df = pd.DataFrame([
            {"ISBN_PT": "1", "Hardcover_Book_ID": 10, "Ano_Publicacao_PT": "2022-05-01", "Flag_Formato_Excluir": False},
            {"ISBN_PT": "2", "Hardcover_Book_ID": 10, "Ano_Publicacao_PT": "2023", "Flag_Formato_Excluir": False},
            {"ISBN_PT": "3", "Hardcover_Book_ID": 20, "Ano_Publicacao_PT": "2010", "Flag_Formato_Excluir": False},
            {"ISBN_PT": "4", "Hardcover_Book_ID": 30, "Ano_Publicacao_PT": "2021", "Flag_Formato_Excluir": True},
            {"ISBN_PT": "5", "Hardcover_Book_ID": None, "Ano_Publicacao_PT": "2021", "Flag_Formato_Excluir": False},
            {"ISBN_PT": "6", "Hardcover_Book_ID": None, "Ano_Publicacao_PT": "2022", "Flag_Formato_Excluir": False},
        ])
        filtrado, meta = filtrar_escopo(df, ano_min=2015)
        self.assertEqual(meta["n_inicial"], 6)
        self.assertEqual(meta["n_antes_ano_min"], 1)
        self.assertEqual(meta["n_formato_excluido"], 1)
        self.assertEqual(len(filtrado), 4)

        ob = deduplicar_por_obra(filtrado)
        # Book 10 deduplicates to 1 row, plus 2 rows with NaN Hardcover_Book_ID = 3 rows
        self.assertEqual(len(ob), 3)

    def test_spice_nivel_and_triangulation(self):
        self.assertEqual(nivel_tag("4 peppers"), 3)
        self.assertEqual(nivel_tag("spice level 3"), 2)
        self.assertEqual(nivel_tag("closed door"), 0)
        self.assertIsNone(nivel_tag("enemies to lovers"))

        cats = {"Tag": [("steamy", 4), ("enemies to lovers", 10)], "Content Warning": [("explicit sex", 2)]}
        sp = spice_da_obra(cats)
        self.assertIn(sp["Spice_Status"], ("positivo",))

        df = pd.DataFrame([
            {
                "ISBN_PT": "9788501",
                "Hardcover_Book_ID": 100,
                "Spice_Nivel": 3,
                "Flag_Spice_Tag": True,
                "Flag_Closed_Door_Tag": False,
                "Tags_Gerais_Hardcover": "spicy | 4 peppers",
                "Generos_Hardcover": "Romance",
                "Moods_Hardcover": "emotional",
                "Content_Warnings": "sexual content",
                "Flag_CW_Sexual_Content": True,
                "Sinopse_PT": "Um dark romance com cenas picantes.",
                "Maturity_GB": "MATURE",
                "Categorias_GB": "Fiction",
                "Tipos_Origem": "autora",
                "Qtd_Usuarios_Hardcover": 500,
            },
            {
                "ISBN_PT": "9788502",
                "Hardcover_Book_ID": 200,
                "Spice_Nivel": None,
                "Flag_Spice_Tag": False,
                "Flag_Closed_Door_Tag": False,
                "Tags_Gerais_Hardcover": "magic",
                "Generos_Hardcover": "Young Adult | Fantasy",
                "Moods_Hardcover": "adventurous",
                "Content_Warnings": "grief",
                "Flag_CW_Sexual_Content": False,
                "Sinopse_PT": "Uma aventura na floresta.",
                "Maturity_GB": "NOT_MATURE",
                "Categorias_GB": "Young Adult Fiction",
                "Tipos_Origem": "autora",
                "Qtd_Usuarios_Hardcover": 600,
            },
            {
                "ISBN_PT": "9788503",
                "Hardcover_Book_ID": 300,
                "Spice_Nivel": None,
                "Flag_Spice_Tag": False,
                "Flag_Closed_Door_Tag": False,
                "Tags_Gerais_Hardcover": "",
                "Generos_Hardcover": "Romance",
                "Moods_Hardcover": "",
                "Content_Warnings": "",
                "Flag_CW_Sexual_Content": False,
                "Sinopse_PT": "Contem conteudo adulto e cenas explicitas.",
                "Maturity_GB": "NOT_MATURE",
                "Categorias_GB": "Fiction",
                "Tipos_Origem": "termo:conteudo",
                "Qtd_Usuarios_Hardcover": 50,
            },
            {
                # Obra sem spice sexual, mas com drogas, bebidas alcoolicas e violencia
                "ISBN_PT": "9788504",
                "Hardcover_Book_ID": 400,
                "Spice_Nivel": None,
                "Flag_Spice_Tag": False,
                "Flag_Closed_Door_Tag": False,
                "Tags_Gerais_Hardcover": "alcoholism | drug use",
                "Generos_Hardcover": "Contemporary",
                "Moods_Hardcover": "dark",
                "Content_Warnings": "violence | alcohol | substance abuse",
                "Flag_CW_Sexual_Content": False,
                "Flag_CW_Violencia": True,
                "Flag_CW_Drogas_Alcool": True,
                "Sinopse_PT": "Uma historia sobre alcoolismo, uso de drogas e violencia extrema.",
                "Maturity_GB": "NOT_MATURE",
                "Categorias_GB": "Fiction",
                "Tipos_Origem": "autora",
                "Qtd_Usuarios_Hardcover": 450,
            },
        ])
        rot = rotular(df, pop_pct=50.0).set_index("Hardcover_Book_ID")
        self.assertEqual(rot.loc[100, "Sensivel_Label"], "alto")
        self.assertEqual(rot.loc[100, "Sensivel_Label_Confianca"], "forte")
        self.assertTrue(bool(rot.loc[100, "Flag_Eixo_Sexo"]))

        self.assertEqual(rot.loc[200, "Sensivel_Label"], "provavel_baixo")

        # Circularity guard: book 300 only has synopsis vote and came from termo:conteudo
        self.assertEqual(rot.loc[300, "Sensivel_Label"], "indeterminado")
        self.assertEqual(rot.loc[300, "Sensivel_Label_Confianca"], "circular")

        # Multi-axis sensitive content (drugs, alcohol, violence) -> Sensivel_Label == 'alto'
        self.assertEqual(rot.loc[400, "Sensivel_Label"], "alto")
        self.assertEqual(rot.loc[400, "Sensivel_Label_Confianca"], "forte")
        self.assertFalse(bool(rot.loc[400, "Flag_Eixo_Sexo"]))
        self.assertTrue(bool(rot.loc[400, "Flag_Eixo_Drogas_Alcool"]))
        self.assertTrue(bool(rot.loc[400, "Flag_Eixo_Violencia"]))
        self.assertEqual(resolver_colunas_rotulo(rot)[0], "Sensivel_Label")

    def test_multi_axis_sensitive_detection(self):
        # 1. montar_obra extrai flags de todos os 4 eixos dos Content Warnings do Hardcover
        book_hc = {
            "id": 999,
            "title": "Dark Nights",
            "contributions": [{"contribution": "Author", "author": {"name": "Jane Doe"}}],
            "cached_tags": {
                "Genre": [{"tag": "Romance", "count": 5}],
                "Content Warning": [
                    {"tag": "alcoholism", "count": 4},
                    {"tag": "drug use", "count": 3},
                    {"tag": "murder", "count": 5},
                    {"tag": "domestic abuse", "count": 2},
                ],
            },
        }
        obra = montar_obra(999, book_hc, min_count_spice=2)
        self.assertFalse(obra["Flag_CW_Sexual_Content"])
        self.assertTrue(obra["Flag_CW_Drogas_Alcool"])
        self.assertEqual(obra["CW_Drogas_Alcool_Contagem"], 7)
        self.assertTrue(obra["Flag_CW_Violencia"])
        self.assertEqual(obra["CW_Violencia_Contagem"], 5)
        self.assertTrue(obra["Flag_CW_Abuso_Gatilhos"])
        self.assertEqual(obra["CW_Abuso_Gatilhos_Contagem"], 2)
        self.assertTrue(obra["Flag_Has_CW"])

        # 2. Votos individuais para alcool/drogas, violencia e abuso em PT e EN
        self.assertEqual(voto_sinopse({"Sinopse_PT": "Ele lutava contra o alcoolismo, bebidas alcoólicas e uso de drogas ilícitas."}), "alto")
        self.assertEqual(voto_sinopse({"Sinopse_PT": "Cenas de violência física, tortura e sequestro."}), "alto")
        self.assertEqual(voto_sinopse({"Sinopse_PT": "Aborda relacionamento abusivo, estupro e automutilação."}), "alto")
        self.assertEqual(voto_vocab(pd.Series({"Tags_Gerais_Hardcover": "drinking | drug abuse", "Generos_Hardcover": ""})), "alto")
        self.assertEqual(voto_cw(pd.Series({"Content_Warnings": "alcohol | gun violence"})), "alto")

        eixos = eixos_sensiveis_obra(pd.Series({
            "Sinopse_PT": "Vício em álcool e relacionamento tóxico.",
            "Content_Warnings": "murder",
            "Tags_Gerais_Hardcover": "",
            "Generos_Hardcover": "",
        }))
        self.assertFalse(eixos["Flag_Eixo_Sexo"])
        self.assertTrue(eixos["Flag_Eixo_Drogas_Alcool"])
        self.assertTrue(eixos["Flag_Eixo_Violencia"])
        self.assertTrue(eixos["Flag_Eixo_Abuso_Gatilhos"])

    def test_eda_stats_helpers(self):
        p, lo, hi = wilson_ic(50, 100)
        self.assertAlmostEqual(p, 0.5)
        self.assertTrue(0.39 < lo < 0.41)
        self.assertTrue(0.59 < hi < 0.61)
        uni, bi = contar_ngramas(["amor secreto na corte sombria", "amor secreto no castelo"])
        self.assertEqual(uni["amor"], 2)
        self.assertEqual(bi["amor secreto"], 2)


class TestLexiconRulesAndEvaluation(unittest.TestCase):
    def test_lexicon_integrity(self):
        from pipeline_ptbr.lexicon import EIXOS, ORDEM_EIXOS, RE_POR_EIXO, tabela_lexico, RE_CW_SENSIVEL
        self.assertEqual(set(EIXOS), set(ORDEM_EIXOS))
        t = tabela_lexico()
        self.assertGreater(len(t), 100)
        self.assertEqual(set(t["Fonte"]), {"cw", "tags", "pt"})
        # 'war' foi removido de violencia de proposito (dispara em fantasia epica)
        self.assertFalse(RE_POR_EIXO["violencia"]["cw"].search("war"))
        self.assertTrue(RE_POR_EIXO["violencia"]["cw"].search("gun violence"))
        # violencia sexual conta em sexo E abuso (decisao de taxonomia documentada)
        self.assertTrue(RE_POR_EIXO["sexo"]["cw"].search("rape"))
        self.assertTrue(RE_POR_EIXO["abuso_gatilhos"]["cw"].search("rape"))
        self.assertTrue(RE_CW_SENSIVEL.search("alcoholism"))
        # labeling e rotulagem devem apontar para o MESMO objeto compilado (sem duplicacao)
        from pipeline_ptbr import labeling, rotulagem
        self.assertIs(labeling.RE_CW_VIOLENCIA, rotulagem.RE_CW_VIOLENCIA)

    def test_rules_table_and_ablation(self):
        from pipeline_ptbr.rotulagem import REGRAS_DECISAO, tabela_regras, decidir
        self.assertEqual([r[0] for r in REGRAS_DECISAO], [f"R{i}" for i in range(1, 9)])
        self.assertEqual(len(tabela_regras()), 8)
        base = {"V_tags": None, "V_vocab": None, "V_sinopse": "alto", "V_maturity": "alto", "V_cw": None,
                "Via_Termo_Conteudo": False, "Cobertura_Boa": True, "YA": True}
        r = pd.Series(base)
        self.assertEqual(decidir(r)[:2], ("alto", "forte"))                         # 2 familias independentes
        self.assertEqual(decidir(r, ignorar=("maturity",))[:2], ("alto", "fraca"))  # ablacao: sobra 1
        self.assertEqual(decidir(r, ignorar=("maturity", "sinopse"))[:2], ("provavel_baixo", "nao_validado"))
        r2 = pd.Series({**base, "V_maturity": None, "Via_Termo_Conteudo": True})
        self.assertEqual(decidir(r2)[1], "circular")
        r3 = pd.Series({**base, "V_vocab": "baixo"})
        self.assertEqual(decidir(r3)[1], "conflito")

    def test_rotular_sensitivity_params(self):
        from pipeline_ptbr.rotulagem import rotular
        df = pd.DataFrame([{
            "ISBN_PT": "1", "Hardcover_Book_ID": 1, "Spice_Nivel": None, "Flag_Spice_Tag": False,
            "Flag_Closed_Door_Tag": False, "Tags_Gerais_Hardcover": "magic", "Generos_Hardcover": "Young Adult",
            "Moods_Hardcover": "fun", "Content_Warnings": "", "Flag_CW_Sexual_Content": False,
            "Sinopse_PT": "Uma aventura.", "Maturity_GB": "NOT_MATURE", "Categorias_GB": "Young Adult Fiction",
            "Tipos_Origem": "autora", "Qtd_Usuarios_Hardcover": 1000,
        }])
        self.assertEqual(rotular(df, 50.0)["Sensivel_Label"].iloc[0], "provavel_baixo")
        self.assertEqual(rotular(df, 50.0, exigir_cw=True)["Sensivel_Label"].iloc[0], "indeterminado")

    def test_kappa_implementations(self):
        from pipeline_ptbr.avaliar_rotulo import cohen_kappa, fleiss_kappa, jaccard
        a = pd.Series(["alto", "alto", "baixo", "baixo", "alto", "baixo"])
        self.assertAlmostEqual(cohen_kappa(a, a)[0], 1.0)
        b = pd.Series(["alto", "baixo", "baixo", "alto", "alto", "baixo"])
        k, n = cohen_kappa(a, b)
        self.assertEqual(n, 6)
        self.assertAlmostEqual(k, 1 / 3, places=6)          # po=4/6, pe=1/2 -> (2/3-1/2)/(1/2)
        tab = pd.DataFrame({"alto": [3, 0, 2], "baixo": [0, 3, 1]})   # 3 avaliadores
        kf, nf = fleiss_kappa(tab)
        self.assertEqual(nf, 3)
        self.assertTrue(0 < kf < 1)
        self.assertAlmostEqual(fleiss_kappa(pd.DataFrame({"alto": [3, 3], "baixo": [0, 0]}))[0], 1.0)
        self.assertAlmostEqual(jaccard(pd.Series([1, 1, 0, 0]), pd.Series([1, 0, 1, 0])), 1 / 3)

    def test_gold_metrics(self):
        from pipeline_ptbr.gold_standard import _metricas_binarias, _norm_resp, cohen_kappa
        self.assertEqual(_norm_resp(pd.Series(["1", "0", "?", "", "S", "n"])).tolist()[:2], [1.0, 0.0])
        self.assertTrue(pd.isna(_norm_resp(pd.Series(["?"])).iloc[0]))
        pred = pd.Series([True, True, False, False, True])
        gold = pd.Series([True, False, False, True, True])
        m = _metricas_binarias(pred, gold)
        self.assertEqual((m["TP"], m["FP"], m["FN"], m["TN"]), (2, 1, 1, 1))
        self.assertAlmostEqual(m["precisao"], 2 / 3)
        self.assertAlmostEqual(m["recall"], 2 / 3)
        k, po, n = cohen_kappa(pd.Series([1.0, 0.0, 1.0, np.nan]), pd.Series([1.0, 0.0, 0.0, 1.0]))
        self.assertEqual(n, 3)


if __name__ == "__main__":
    unittest.main()
