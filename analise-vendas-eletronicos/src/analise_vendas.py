"""
Análise de Vendas de Produtos Eletrônicos
=========================================

Projeto de análise exploratória de dados usando:
    - pandas     -> leitura, limpeza e agregações
    - numpy      -> cálculos numéricos e estatísticos
    - matplotlib -> visualização dos resultados

Uso:
    python src/analise_vendas.py                 # analisa e mostra os gráficos
    python src/analise_vendas.py --nao-mostrar   # analisa e só salva os gráficos
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_CSV = RAIZ / "data" / "vendas_eletronicos.csv"
PASTA_GRAFICOS = RAIZ / "outputs" / "graficos"

MESES_PT = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
            "Jul", "Ago", "Set", "Out", "Nov", "Dez"]

# Paleta de cores consistente em todos os gráficos
AZUL = "#1F6FEB"
AZUL_CLARO = "#8DB6F5"
LARANJA = "#F28C28"
CINZA = "#6B7280"
PALETA = ["#1F6FEB", "#F28C28", "#2DA44E", "#8250DF", "#CF222E", "#6B7280"]

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlepad": 12,
    "font.size": 10,
})


# ---------------------------------------------------------------------------
# Funções auxiliares de formatação (padrão brasileiro)
# ---------------------------------------------------------------------------
def moeda(valor: float) -> str:
    """Formata número como moeda brasileira: 1234567.8 -> 'R$ 1.234.567,80'."""
    texto = f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {texto}"


def moeda_curta(valor: float, _pos=None) -> str:
    """Versão curta para eixos: 1_500_000 -> 'R$ 1,5 mi'."""
    if abs(valor) >= 1_000_000:
        return f"R$ {valor / 1_000_000:.1f} mi".replace(".", ",")
    if abs(valor) >= 1_000:
        return f"R$ {valor / 1_000:.0f} mil"
    return f"R$ {valor:.0f}"


# ---------------------------------------------------------------------------
# 1. Carga e limpeza dos dados
# ---------------------------------------------------------------------------
def carregar_dados(caminho: Path = CAMINHO_CSV) -> pd.DataFrame:
    """Lê o CSV e converte a coluna de data."""
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}\n"
            "Rode primeiro: python src/gerar_dados.py"
        )
    return pd.read_csv(caminho, parse_dates=["data_venda"])


def diagnostico(df: pd.DataFrame) -> None:
    """Imprime um resumo rápido da qualidade dos dados."""
    print("=" * 60)
    print("1. DIAGNÓSTICO DOS DADOS")
    print("=" * 60)
    print(f"Linhas: {len(df)} | Colunas: {df.shape[1]}")
    print(f"Período: {df['data_venda'].min():%d/%m/%Y} a {df['data_venda'].max():%d/%m/%Y}")
    print(f"Linhas duplicadas: {df.duplicated().sum()}")
    nulos = df.isna().sum()
    print("Valores nulos por coluna:")
    print(nulos[nulos > 0].to_string() if nulos.any() else "  nenhum")
    print()


def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicatas e trata valores nulos."""
    antes = len(df)
    df = df.drop_duplicates().copy()
    print(f"Duplicatas removidas: {antes - len(df)}")

    n_nulos = df["forma_pagamento"].isna().sum()
    df["forma_pagamento"] = df["forma_pagamento"].fillna("Não informado")
    print(f"Nulos em 'forma_pagamento' preenchidos: {n_nulos}\n")
    return df


def criar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Cria colunas calculadas usadas nas análises."""
    df["receita_bruta"] = df["preco_unitario"] * df["quantidade"]
    df["valor_desconto"] = df["receita_bruta"] * df["desconto"]
    df["receita_liquida"] = df["receita_bruta"] - df["valor_desconto"]

    df["ano"] = df["data_venda"].dt.year
    df["mes"] = df["data_venda"].dt.month
    df["ano_mes"] = df["data_venda"].dt.to_period("M")
    df["dia_semana"] = df["data_venda"].dt.dayofweek  # 0 = segunda

    # np.select: classifica cada venda em faixas de desconto
    condicoes = [df["desconto"] == 0, df["desconto"] <= 0.10]
    faixas = ["Sem desconto", "Até 10%"]
    df["faixa_desconto"] = np.select(condicoes, faixas, default="Acima de 10%")
    return df


# ---------------------------------------------------------------------------
# 2. Análises
# ---------------------------------------------------------------------------
def calcular_kpis(df: pd.DataFrame) -> dict:
    """Indicadores principais do negócio."""
    return {
        "receita_liquida": df["receita_liquida"].sum(),
        "total_vendas": len(df),
        "unidades": int(df["quantidade"].sum()),
        "ticket_medio": df["receita_liquida"].mean(),
        "mediana_ticket": float(np.median(df["receita_liquida"])),
        "desconto_total": df["valor_desconto"].sum(),
    }


def receita_por_categoria(df: pd.DataFrame) -> pd.Series:
    return df.groupby("categoria")["receita_liquida"].sum().sort_values(ascending=False)


def top_produtos(df: pd.DataFrame, n: int = 10) -> pd.Series:
    return df.groupby("produto")["receita_liquida"].sum().nlargest(n)


def receita_mensal(df: pd.DataFrame) -> pd.Series:
    return df.groupby("ano_mes")["receita_liquida"].sum()


def receita_por_regiao(df: pd.DataFrame) -> pd.Series:
    return df.groupby("regiao")["receita_liquida"].sum().sort_values(ascending=False)


def receita_por_pagamento(df: pd.DataFrame) -> pd.Series:
    return df.groupby("forma_pagamento")["receita_liquida"].sum().sort_values(ascending=False)


def unidades_por_faixa_desconto(df: pd.DataFrame) -> pd.Series:
    ordem = ["Sem desconto", "Até 10%", "Acima de 10%"]
    return df.groupby("faixa_desconto")["quantidade"].mean().reindex(ordem)


def media_movel(valores: np.ndarray, janela: int = 3) -> np.ndarray:
    """Média móvel simples usando numpy (np.convolve)."""
    kernel = np.ones(janela) / janela
    return np.convolve(valores, kernel, mode="valid")


def analises_estatisticas(df: pd.DataFrame) -> dict:
    """Análises com numpy: correlação, outliers e crescimento anual."""
    # Correlação entre desconto e quantidade comprada
    corr = np.corrcoef(df["desconto"].to_numpy(), df["quantidade"].to_numpy())[0, 1]

    # Outliers de receita por venda (método do IQR)
    q1, q3 = np.percentile(df["receita_liquida"], [25, 75])
    limite_superior = q3 + 1.5 * (q3 - q1)
    outliers = int((df["receita_liquida"] > limite_superior).sum())

    # Crescimento 2025 vs 2024
    rec_ano = df.groupby("ano")["receita_liquida"].sum()
    crescimento = np.nan
    if {2024, 2025}.issubset(rec_ano.index):
        crescimento = (rec_ano[2025] / rec_ano[2024] - 1) * 100

    return {
        "correlacao_desconto_qtd": corr,
        "limite_outlier": limite_superior,
        "qtd_outliers": outliers,
        "crescimento_anual_pct": crescimento,
    }


def imprimir_resultados(df, kpis, stats) -> None:
    print("=" * 60)
    print("2. INDICADORES PRINCIPAIS (KPIs)")
    print("=" * 60)
    print(f"Receita líquida total : {moeda(kpis['receita_liquida'])}")
    print(f"Número de vendas      : {kpis['total_vendas']:,}".replace(",", "."))
    print(f"Unidades vendidas     : {kpis['unidades']:,}".replace(",", "."))
    print(f"Ticket médio          : {moeda(kpis['ticket_medio'])}")
    print(f"Mediana do ticket     : {moeda(kpis['mediana_ticket'])}")
    print(f"Total em descontos    : {moeda(kpis['desconto_total'])}")
    print()

    print("=" * 60)
    print("3. RANKINGS")
    print("=" * 60)
    print("Receita por categoria:")
    for nome, valor in receita_por_categoria(df).items():
        print(f"  {nome:<12} {moeda(valor):>18}")
    print("\nTop 5 produtos:")
    for nome, valor in top_produtos(df, 5).items():
        print(f"  {nome:<28} {moeda(valor):>18}")
    print()

    mensal = receita_mensal(df)
    melhor_mes = mensal.idxmax()
    print(f"Melhor mês: {MESES_PT[melhor_mes.month - 1]}/{melhor_mes.year} "
          f"({moeda(mensal.max())})")
    print()

    print("=" * 60)
    print("4. ESTATÍSTICAS (numpy)")
    print("=" * 60)
    print(f"Correlação desconto x quantidade: {stats['correlacao_desconto_qtd']:.3f}")
    print(f"Vendas outliers (acima de {moeda(stats['limite_outlier'])}): {stats['qtd_outliers']}")
    if not np.isnan(stats["crescimento_anual_pct"]):
        print(f"Crescimento da receita 2025 vs 2024: {stats['crescimento_anual_pct']:+.1f}%")
    print()


# ---------------------------------------------------------------------------
# 3. Gráficos (cada função desenha em um "ax" para reaproveitar no dashboard)
# ---------------------------------------------------------------------------
def grafico_categorias(ax, df):
    dados = receita_por_categoria(df).sort_values()
    barras = ax.barh(dados.index, dados.values, color=AZUL)
    ax.set_xticks([])  # valores já aparecem nos rótulos das barras
    ax.set_title("Receita por categoria")
    ax.grid(visible=False)
    ax.bar_label(barras, labels=[moeda_curta(v) for v in dados.values], padding=4, fontsize=8)
    ax.set_xlim(0, dados.max() * 1.18)


def grafico_evolucao_mensal(ax, df):
    mensal = receita_mensal(df)
    x = np.arange(len(mensal))
    rotulos = [f"{MESES_PT[p.month - 1]}\n{str(p.year)[2:]}" if p.month in (1, 4, 7, 10) else ""
               for p in mensal.index]

    ax.plot(x, mensal.values, color=AZUL_CLARO, marker="o", markersize=4, label="Receita mensal")
    mm = media_movel(mensal.to_numpy(), janela=3)
    ax.plot(x[2:], mm, color=LARANJA, linewidth=2.5, label="Média móvel (3 meses)")

    # Destaca o pico de vendas
    i_max = int(np.argmax(mensal.values))
    ax.annotate(
        f"Pico: {MESES_PT[mensal.index[i_max].month - 1]}/{mensal.index[i_max].year}",
        xy=(i_max, mensal.values[i_max]),
        xytext=(i_max - 5, mensal.values[i_max] * 0.95),
        arrowprops=dict(arrowstyle="->", color=CINZA), fontsize=9, color=CINZA,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(rotulos)
    ax.yaxis.set_major_formatter(FuncFormatter(moeda_curta))
    ax.set_title("Evolução mensal da receita")
    ax.legend(frameon=False, loc="upper left")


def grafico_top_produtos(ax, df):
    dados = top_produtos(df, 8).sort_values()
    barras = ax.barh(dados.index, dados.values, color=AZUL)
    ax.set_xticks([])
    ax.set_title("Top 8 produtos por receita")
    ax.grid(visible=False)
    ax.bar_label(barras, labels=[moeda_curta(v) for v in dados.values], padding=4, fontsize=8)
    ax.set_xlim(0, dados.max() * 1.18)


def grafico_regioes(ax, df):
    dados = receita_por_regiao(df)
    barras = ax.bar(dados.index, dados.values, color=PALETA[: len(dados)])
    ax.yaxis.set_major_formatter(FuncFormatter(moeda_curta))
    ax.set_title("Receita por região")
    ax.grid(axis="x", visible=False)
    ax.bar_label(barras, labels=[f"{v / dados.sum() * 100:.0f}%" for v in dados.values],
                 padding=3, fontsize=9)
    ax.set_ylim(0, dados.max() * 1.12)
    ax.tick_params(axis="x", labelsize=8)


def grafico_pagamento(ax, df):
    dados = receita_por_pagamento(df)
    percentuais = dados / dados.sum() * 100
    # Fatias < 2% (ex.: "Não informado") ficam sem rótulo para não poluir
    rotulos = [nome if pct >= 2 else "" for nome, pct in percentuais.items()]
    ax.pie(
        dados.values, labels=rotulos, colors=PALETA[: len(dados)],
        autopct=lambda p: f"{p:.0f}%" if p >= 2 else "",
        startangle=90, pctdistance=0.78,
        wedgeprops=dict(width=0.42, edgecolor="white"),
        textprops=dict(fontsize=8),
    )
    ax.set_title("Formas de pagamento")
    ax.grid(False)


def grafico_desconto(ax, df):
    dados = unidades_por_faixa_desconto(df)
    barras = ax.bar(dados.index, dados.values, color=[AZUL_CLARO, AZUL, LARANJA])
    ax.set_title("Desconto x unidades por venda")
    ax.set_ylabel("Média de unidades por venda")
    ax.grid(axis="x", visible=False)
    ax.bar_label(barras, fmt="%.2f", padding=3, fontsize=9)
    ax.set_ylim(0, dados.max() * 1.15)


GRAFICOS = [
    ("01_receita_por_categoria", grafico_categorias),
    ("02_evolucao_mensal", grafico_evolucao_mensal),
    ("03_top_produtos", grafico_top_produtos),
    ("04_receita_por_regiao", grafico_regioes),
    ("05_formas_pagamento", grafico_pagamento),
    ("06_desconto_vs_unidades", grafico_desconto),
]


def salvar_graficos_individuais(df) -> None:
    PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)
    for nome, funcao in GRAFICOS:
        fig, ax = plt.subplots(figsize=(8, 5))
        funcao(ax, df)
        fig.tight_layout()
        fig.savefig(PASTA_GRAFICOS / f"{nome}.png", dpi=150)
        plt.close(fig)


def salvar_dashboard(df, kpis):
    """Painel único com todos os gráficos (ótimo para postar no LinkedIn)."""
    fig = plt.figure(figsize=(16, 10.5))
    fig.suptitle("Dashboard de Vendas - Produtos Eletrônicos (2024-2025)",
                 fontsize=20, fontweight="bold", y=0.985)

    # Faixa de KPIs no topo
    textos_kpi = [
        ("Receita líquida", moeda(kpis["receita_liquida"])),
        ("Vendas", f"{kpis['total_vendas']:,}".replace(",", ".")),
        ("Unidades", f"{kpis['unidades']:,}".replace(",", ".")),
        ("Ticket médio", moeda(kpis["ticket_medio"])),
    ]
    for i, (rotulo, valor) in enumerate(textos_kpi):
        x = 0.14 + i * 0.24
        fig.text(x, 0.925, valor, ha="center", fontsize=17, fontweight="bold", color=AZUL)
        fig.text(x, 0.900, rotulo, ha="center", fontsize=10, color=CINZA)

    grade = fig.add_gridspec(2, 3, top=0.82, bottom=0.07, left=0.08, right=0.97,
                             hspace=0.40, wspace=0.35)
    # O gráfico mensal ocupa 2 colunas da primeira linha
    layout = [
        (grade[0, 0:2], grafico_evolucao_mensal),
        (grade[0, 2], grafico_pagamento),
        (grade[1, 0], grafico_categorias),
        (grade[1, 1], grafico_top_produtos),
        (grade[1, 2], grafico_regioes),
    ]
    for posicao, funcao in layout:
        funcao(fig.add_subplot(posicao), df)

    PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PASTA_GRAFICOS / "00_dashboard.png", dpi=150, facecolor="white")
    return fig


# ---------------------------------------------------------------------------
# Execução principal
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Análise de vendas de eletrônicos")
    parser.add_argument("--nao-mostrar", action="store_true",
                        help="apenas salva os gráficos, sem abrir janelas")
    args = parser.parse_args()

    df = carregar_dados()
    diagnostico(df)
    df = limpar_dados(df)
    df = criar_colunas(df)

    kpis = calcular_kpis(df)
    stats = analises_estatisticas(df)
    imprimir_resultados(df, kpis, stats)

    salvar_graficos_individuais(df)
    fig = salvar_dashboard(df, kpis)
    print(f"📊 Gráficos salvos em: {PASTA_GRAFICOS}")

    if not args.nao_mostrar:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
