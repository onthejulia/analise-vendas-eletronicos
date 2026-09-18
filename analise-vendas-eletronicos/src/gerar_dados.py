from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_CSV = RAIZ / "data" / "vendas_eletronicos.csv"

N_VENDAS = 3000
SEMENTE = 42  # semente fixa = resultados reproduzíveis

# (produto, categoria, marca, preço base em R$, popularidade)
CATALOGO = [
    ("iPhone 15", "Smartphones", "Apple", 5499, 6),
    ("Galaxy S24", "Smartphones", "Samsung", 4599, 8),
    ("Moto G84", "Smartphones", "Motorola", 1499, 12),
    ("Redmi Note 13", "Smartphones", "Xiaomi", 1699, 10),
    ("MacBook Air M2", "Notebooks", "Apple", 8999, 3),
    ("Inspiron 15", "Notebooks", "Dell", 3499, 6),
    ("Ideapad 3", "Notebooks", "Lenovo", 2899, 8),
    ("Aspire 5", "Notebooks", "Acer", 2999, 5),
    ("Smart TV 50\" 4K", "TVs", "Samsung", 2799, 9),
    ("Smart TV 55\" OLED", "TVs", "LG", 5999, 3),
    ("Smart TV 43\" Full HD", "TVs", "TCL", 1699, 8),
    ("PlayStation 5", "Games", "Sony", 3999, 5),
    ("Xbox Series S", "Games", "Microsoft", 2499, 4),
    ("Nintendo Switch", "Games", "Nintendo", 2699, 4),
    ("AirPods Pro", "Áudio", "Apple", 1899, 8),
    ("JBL Charge 5", "Áudio", "JBL", 999, 10),
    ("Fone Bluetooth Redmi Buds", "Áudio", "Xiaomi", 199, 15),
    ("Apple Watch SE", "Wearables", "Apple", 2499, 4),
    ("Galaxy Watch 6", "Wearables", "Samsung", 1799, 4),
    ("Mi Band 8", "Wearables", "Xiaomi", 299, 12),
    ("Carregador Turbo 65W", "Acessórios", "Anker", 149, 18),
    ("Cabo USB-C 2m", "Acessórios", "Baseus", 39, 25),
    ("Mouse Sem Fio", "Acessórios", "Logitech", 129, 16),
    ("Teclado Mecânico", "Acessórios", "Redragon", 249, 10),
]

# Estados por região, com peso (SP e RJ vendem mais, por exemplo)
REGIOES = {
    "Sudeste": (0.45, [("SP", 0.55), ("RJ", 0.20), ("MG", 0.20), ("ES", 0.05)]),
    "Sul": (0.18, [("PR", 0.40), ("RS", 0.35), ("SC", 0.25)]),
    "Nordeste": (0.20, [("BA", 0.30), ("PE", 0.25), ("CE", 0.25), ("PI", 0.20)]),
    "Centro-Oeste": (0.10, [("GO", 0.45), ("DF", 0.35), ("MT", 0.20)]),
    "Norte": (0.07, [("PA", 0.50), ("AM", 0.35), ("TO", 0.15)]),
}

# Multiplicador de vendas por mês (sazonalidade)
PESO_MES = {1: 0.80, 2: 0.75, 3: 0.90, 4: 0.90, 5: 1.10, 6: 1.00,
            7: 0.95, 8: 0.90, 9: 0.95, 10: 1.00, 11: 1.90, 12: 1.70}


def gerar_vendas() -> pd.DataFrame:
    rng = np.random.default_rng(SEMENTE)

    # ---- Datas com sazonalidade ------------------------------------------
    datas_possiveis = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    pesos_dia = np.array([PESO_MES[d.month] for d in datas_possiveis])
    # 2025 cresce ~12% em relação a 2024
    pesos_dia = pesos_dia * np.where(datas_possiveis.year == 2025, 1.12, 1.0)
    pesos_dia = pesos_dia / pesos_dia.sum()
    datas = rng.choice(datas_possiveis, size=N_VENDAS, p=pesos_dia)

    # ---- Produtos (mais populares aparecem mais) -------------------------
    popularidade = np.array([item[4] for item in CATALOGO], dtype=float)
    idx_produtos = rng.choice(len(CATALOGO), size=N_VENDAS, p=popularidade / popularidade.sum())
    produtos = np.array([CATALOGO[i][0] for i in idx_produtos])
    categorias = np.array([CATALOGO[i][1] for i in idx_produtos])
    marcas = np.array([CATALOGO[i][2] for i in idx_produtos])
    preco_base = np.array([CATALOGO[i][3] for i in idx_produtos], dtype=float)

    # Preço varia ±5% em torno do preço base
    preco_unitario = np.round(preco_base * rng.uniform(0.95, 1.05, N_VENDAS), 2)

    # ---- Quantidade: produtos baratos vendem em maior quantidade ---------
    qtd_barato = rng.choice([1, 2, 3, 4, 5], size=N_VENDAS, p=[0.45, 0.25, 0.15, 0.10, 0.05])
    qtd_caro = rng.choice([1, 2, 3], size=N_VENDAS, p=[0.85, 0.11, 0.04])
    quantidade = np.where(preco_base < 300, qtd_barato, qtd_caro)

    # ---- Desconto: Black Friday (novembro) tem descontos maiores ---------
    meses = pd.DatetimeIndex(datas).month.to_numpy()
    opcoes_desc = [0.00, 0.05, 0.10, 0.15, 0.20]
    desc_normal = rng.choice(opcoes_desc, size=N_VENDAS, p=[0.55, 0.20, 0.15, 0.07, 0.03])
    desc_bf = rng.choice(opcoes_desc, size=N_VENDAS, p=[0.10, 0.15, 0.25, 0.30, 0.20])
    desconto = np.where(meses == 11, desc_bf, desc_normal)

    # Quanto maior o desconto, maior a chance de o cliente levar +1 unidade
    compra_extra = rng.random(N_VENDAS) < desconto * 1.5
    quantidade = np.minimum(quantidade + compra_extra, 5)

    # ---- Região e estado --------------------------------------------------
    nomes_regioes = list(REGIOES.keys())
    pesos_regioes = np.array([REGIOES[r][0] for r in nomes_regioes])
    regiao = rng.choice(nomes_regioes, size=N_VENDAS, p=pesos_regioes / pesos_regioes.sum())
    estado = np.empty(N_VENDAS, dtype=object)
    for nome in nomes_regioes:
        mascara = regiao == nome
        ufs = [u for u, _ in REGIOES[nome][1]]
        pesos_ufs = np.array([p for _, p in REGIOES[nome][1]])
        estado[mascara] = rng.choice(ufs, size=mascara.sum(), p=pesos_ufs / pesos_ufs.sum())

    # ---- Pagamento e canal ------------------------------------------------
    forma_pagamento = rng.choice(
        ["Cartão de Crédito", "Pix", "Boleto", "Cartão de Débito"],
        size=N_VENDAS, p=[0.50, 0.30, 0.08, 0.12],
    )
    canal = rng.choice(["Online", "Loja Física"], size=N_VENDAS, p=[0.62, 0.38])

    df = pd.DataFrame({
        "data_venda": pd.to_datetime(datas),
        "produto": produtos,
        "categoria": categorias,
        "marca": marcas,
        "preco_unitario": preco_unitario,
        "quantidade": quantidade,
        "desconto": desconto,
        "regiao": regiao,
        "estado": estado,
        "forma_pagamento": forma_pagamento,
        "canal_venda": canal,
    })

    df = df.sort_values("data_venda").reset_index(drop=True)
    df.insert(0, "id_venda", np.arange(1, len(df) + 1))
    df["data_venda"] = df["data_venda"].dt.strftime("%Y-%m-%d")

    # ---- "Sujeira" proposital para praticar limpeza de dados -------------
    nulos = rng.choice(df.index, size=15, replace=False)
    df.loc[nulos, "forma_pagamento"] = np.nan

    duplicadas = df.sample(n=10, random_state=SEMENTE)
    df = pd.concat([df, duplicadas], ignore_index=True)

    return df


def main() -> None:
    CAMINHO_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = gerar_vendas()
    df.to_csv(CAMINHO_CSV, index=False, encoding="utf-8")
    print(f"✅ CSV gerado com {len(df)} linhas em: {CAMINHO_CSV}")


if __name__ == "__main__":
    main()
