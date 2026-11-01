"""
diagnostico_rotulo.py  (colocar em scripts/pipeline_ptbr/)

Diagnóstico do rótulo triangulado (rodar DEPOIS de rotulagem.py). Responde, com os seus dados:
  1. De onde vem cada 'alto' (combinações de fontes) e cada 'baixo' (tem evidência própria ou só a regra YA?)
  2. Circularidade: o voto de sinopse 'alto' concentra-se nas obras que entraram no seed via 'termo:conteudo'?
     (a busca por "conteúdo adulto" casa com a sinopse por construção) + distribuição do rótulo SEM essas obras
  3. Recall relativo das fontes tomando as tags do Hardcover como pseudo-referência (alta precisão, n pequeno)
  4. Vocabulário de spice que o labeling.py pode estar perdendo (ex.: '4 peppers' em Tags_Gerais)
  5. Lista de revisão manual: obras 'baixo' em estratos que esperamos com spice (new_adult, dark, romcom, esportivo)

Saídas: outputs/diagnostico_rotulo.txt, outputs/diagnostico_baixo_revisar.csv, outputs/diagnostico_peppers.csv

Uso: python scripts/pipeline_ptbr/diagnostico_rotulo.py [--ano-min 2015] [--entrada CSV]
"""
import re
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
ENTRADA = RAIZ / "data" / "processed" / "livros_ptbr_v3_rotulado.csv"
OUT = RAIZ / "outputs"

RE_PEPPER = re.compile(r"(\d)\s*peppers?")
RE_SPICE_LIVRE = re.compile(r"spic|steam|open door|smut|explicit|heat|erotic|pepper|sex|hot")
ESTRATOS_ESPERA_SPICE = ('new_adult', 'dark', 'romcom', 'esportivo', 'romantasia')


def _b(x) -> bool:
    return str(x).strip().lower() in ("true", "1", "1.0")


def carregar(path: Path, ano_min: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    ano = df['Ano_Publicacao_PT'].astype(str).str.extract(r"(\d{4})")[0].astype(float)
    df = df[ano >= ano_min]
    if 'Flag_Formato_Excluir' in df.columns:
        df = df[~df['Flag_Formato_Excluir'].map(_b)]
    # nível obra (edições sem ID ficam individuais), mesma lógica da EDA
    sem_id = df[df['Hardcover_Book_ID'].isna()]
    com_id = df[df['Hardcover_Book_ID'].notna()].drop_duplicates('Hardcover_Book_ID')
    return pd.concat([com_id, sem_id]).reset_index(drop=True)


def tags_de(s) -> list:
    return [t.strip().lower() for t in str(s).split('|') if t.strip() and str(s) != 'nan']


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--entrada', type=Path, default=ENTRADA)
    ap.add_argument('--ano-min', type=int, default=2015)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    ob = carregar(a.entrada, a.ano_min)
    L = []
    w = L.append
    w(f"DIAGNÓSTICO DO RÓTULO | obras: {len(ob)} (ano >= {a.ano_min}, sem formatos excluídos)\n" + "=" * 90)

    # ---- 1. composição
    w("\n1) COMPOSIÇÃO DO RÓTULO POR COMBINAÇÃO DE FONTES")
    comp = ob.groupby(['Spice_Label', ob['Spice_Label_Fontes'].fillna('')]).size()
    w(comp.to_string())
    nb = int((ob['Spice_Label'] == 'baixo').sum())
    so_ya = int(((ob['Spice_Label'] == 'baixo') & (ob['Spice_Label_Fontes'] == 'cobertura_ya')).sum())
    w(f"\n  'baixo' sem NENHUM voto próprio (só regra cobertura+YA): {so_ya}/{nb}")
    w("  Se for ~100%, 'baixo' é uma INFERÊNCIA a partir do gênero YA, não uma medição: não use como negativo "
      "sem validação humana (e evite o atalho 'capa fofa/YA => baixo' no POC II).")

    # ---- 2. circularidade com o canal de seed
    w("\n2) CIRCULARIDADE COM O CANAL 'termo:conteudo' (seed buscou por termos de conteúdo adulto)")
    via = ob['Tipos_Origem'].fillna('').str.contains('termo:conteudo') if 'Tipos_Origem' in ob.columns else pd.Series(False, index=ob.index)
    w(f"  Obras que entraram via termo:conteudo: {int(via.sum())}/{len(ob)}")
    w("  V_sinopse x via_termo_conteudo:")
    w("    " + pd.crosstab(ob['V_sinopse'].fillna('abstem'), via.rename('via_conteudo')).to_string().replace("\n", "\n    "))
    w("  Spice_Label x via_termo_conteudo:")
    w("    " + pd.crosstab(ob['Spice_Label'], via.rename('via_conteudo')).to_string().replace("\n", "\n    "))
    w("  Spice_Label SEM as obras via termo:conteudo (análise de sensibilidade):")
    w("    " + ob.loc[~via, 'Spice_Label'].value_counts().to_string().replace("\n", "\n    "))

    # ---- 3. recall relativo (tags como pseudo-referência)
    w("\n3) RECALL RELATIVO: entre obras com V_tags=alto, % em que cada outra fonte também vota alto")
    ref = ob[ob['V_tags'] == 'alto']
    w(f"  n (referência, V_tags=alto) = {len(ref)}  [n pequeno: leia como ordem de grandeza]")
    for f in ('V_sinopse', 'V_maturity', 'V_cw'):
        k = int((ref[f] == 'alto').sum())
        w(f"  {f:<11} {k}/{len(ref)} = {100 * k / max(len(ref), 1):.0f}%")
    w("  Recall baixo da sinopse/maturity = o léxico/fonte perde muita coisa; o 'indeterminado' esconde spice.")

    # ---- 4. vocabulário de spice não capturado
    w("\n4) TAGS COM VOCABULÁRIO DE SPICE em Tags_Gerais/Generos (o que o labeling.py pode estar perdendo)")
    cont = Counter()
    for col in ('Tags_Gerais_Hardcover', 'Generos_Hardcover', 'Moods_Hardcover'):
        if col in ob.columns:
            for s in ob[col]:
                for t in set(tags_de(s)):
                    if RE_SPICE_LIVRE.search(t):
                        cont[t] += 1
    w("  " + (", ".join(f"{t} ({n})" for t, n in cont.most_common(40)) or "(nenhuma)"))
    linhas = []
    for _, r in ob.iterrows():
        for t in tags_de(r.get('Tags_Gerais_Hardcover')):
            m = RE_PEPPER.search(t)
            if m:
                linhas.append({'Titulo_PT': r.get('Titulo_PT'), 'Autor_PT': r.get('Autor_PT'), 'Tag': t,
                               'Peppers': int(m.group(1)), 'V_tags': r.get('V_tags'), 'Spice_Label': r.get('Spice_Label')})
    pep = pd.DataFrame(linhas)
    pep.to_csv(OUT / 'diagnostico_peppers.csv', index=False, encoding='utf-8')
    w(f"\n  Obras com tag 'N peppers': {len(pep)} (diagnostico_peppers.csv). Cruze com Spice_Label: se 4-5 peppers "
      f"aparecem como 'indeterminado' ou 'baixo', o labeling.py não está lendo essa escala.")
    if len(pep):
        w("    " + pd.crosstab(pep['Peppers'], pep['Spice_Label']).to_string().replace("\n", "\n    "))

    # ---- 5. revisão manual dos 'baixo' suspeitos
    w("\n5) 'BAIXO' EM ESTRATOS QUE ESPERAMOS COM SPICE (revisar à mão)")
    est = ob['Estratos_Origem'].fillna('')
    sus = ob[(ob['Spice_Label'] == 'baixo') & est.apply(lambda s: any(e in s for e in ESTRATOS_ESPERA_SPICE))]
    cols = [c for c in ('Titulo_PT', 'Autor_PT', 'Estratos_Origem', 'Generos_Hardcover', 'Categorias_GB',
                        'Maturity_GB', 'Qtd_Usuarios_Hardcover', 'Spice_Label_Fontes') if c in sus.columns]
    sus[cols].to_csv(OUT / 'diagnostico_baixo_revisar.csv', index=False, encoding='utf-8')
    w(f"  {len(sus)} obras em diagnostico_baixo_revisar.csv. Cada falso 'baixo' aqui é evidência de que a regra YA erra.")

    txt = "\n".join(L)
    (OUT / 'diagnostico_rotulo.txt').write_text(txt, encoding='utf-8')
    print(txt)


if __name__ == "__main__":
    main()