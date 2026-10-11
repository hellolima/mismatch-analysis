"""Configuracao central: caminhos, tokens e hiperparametros. Os modulos leem `config.X` em tempo de uso,
entao o run.py pode sobrescrever valores a partir da linha de comando."""
import os
import socket

import urllib3.util.connection as urllib3_cn
from dotenv import load_dotenv

# Forca IPv4 (IPv6 instavel causa travamento silencioso)
urllib3_cn.allowed_gai_family = lambda: socket.AF_INET

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.dirname(PKG_DIR)                 # .../scripts
ROOT = os.path.dirname(SCRIPT_DIR)                    # raiz do projeto

load_dotenv(os.path.join(ROOT, ".env"))
load_dotenv()

HC_TOKEN = (os.getenv("HARDCOVER_API_TOKEN") or "").strip()
GB_KEY = os.getenv("GOOGLE_BOOKS_API_KEY")
HC_URL = "https://api.hardcover.app/v1/graphql"
GB_URL = "https://www.googleapis.com/books/v1/volumes"

SEEDS_DIR = os.path.join(ROOT, "data", "seeds")
PROC_DIR = os.path.join(ROOT, "data", "processed")
COVERS_DIR = os.path.join(ROOT, "data", "covers")
CACHE_DIR = os.path.join(ROOT, "data", "cache")

# listas de entrada
AUTORES_TXT = os.path.join(SEEDS_DIR, "autores_v2.txt")
TERMOS_TXT = os.path.join(SEEDS_DIR, "termos_v2.txt")
EDITORAS_TXT = os.path.join(SEEDS_DIR, "editoras.txt")
TITULOS_TXT = os.path.join(SEEDS_DIR, "titulos.txt")

# estado do seed
SEED_CSV = os.path.join(SEEDS_DIR, "seed_gbooks_ptbr_v3.csv")
SEED_V2_CSV = os.path.join(SEEDS_DIR, "seed_gbooks_ptbr.csv")       # migrado automaticamente se existir
STATE_JSONL = os.path.join(SEEDS_DIR, "queries_done.jsonl")

# caches da ponte (tornam a ponte incremental: so processa ISBNs novos)
CACHE_PONTE = os.path.join(CACHE_DIR, "ponte_cache.json")
CACHE_OBRAS = os.path.join(CACHE_DIR, "obras_cache.json")
CACHE_CATALOGO = os.path.join(CACHE_DIR, "catalogo_autora.json")

# saidas
OUT_EDICOES = os.path.join(PROC_DIR, "edicoes_ptbr_v3.csv")
OUT_OBRAS = os.path.join(PROC_DIR, "obras_ptbr_v3.csv")
OUT_LIVROS = os.path.join(PROC_DIR, "livros_ptbr_v3.csv")
NAO_RES_CSV = os.path.join(PROC_DIR, "nao_resolvidos.csv")
AUDITORIA_CSV = os.path.join(PROC_DIR, "auditoria_ponte.csv")
CAPAS_MANIFEST = os.path.join(PROC_DIR, "capas_manifest.csv")
EDA_SCRIPT = os.path.join(SCRIPT_DIR, "02_eda_booktok_ptbr.py")

# Hiperparametros da ponte (CALIBRAR contra o Gold Standard)
FUZZY_ACEITE = 0.90
EMB_ACEITE = 0.75
EMB_MARGEM = 0.08
MODELO_EMB = "paraphrase-multilingual-MiniLM-L12-v2"
REVISAR_ABAIXO = 0.95
REVISAR_TODO_FALLBACK = True      # True: todo match que nao veio de ISBN entra na auditoria


def garantir_dirs():
    for d in (SEEDS_DIR, PROC_DIR, COVERS_DIR, CACHE_DIR):
        os.makedirs(d, exist_ok=True)
