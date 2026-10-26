"""
Etapa de Aquisição e Engenharia de Features (yfinance)
Baixamos o histórico das 5 principais ações da B3 usando o yahoo finance (que é aberto), calculamos volatilidades 
realizadas em várias escalas e o alvo de volatilidade futura
e salvamos o dataset processado localmente na pasta data/ usando caminhos relativos.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf 

# =====================================================================
# 1. CAMINHOS RELATIVOS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV_PATH = DATA_DIR / "b3_stocks_raw.csv"
FEATURES_CSV_PATH = DATA_DIR / "b3_stocks_features.csv"
SUMMARY_CSV_PATH = DATA_DIR / "b3_summary_stats.csv"

# =====================================================================
# 2. DATASET
# =====================================================================

# As 5 ações de maior peso/liquidez na B3
TICKERS = [
    "PETR4.SA",  # Petrobras
    "VALE3.SA",  # Vale
    "ITUB4.SA",  # Itaú Unibanco
    "BBDC4.SA",  # Bradesco
    "WEGE3.SA",  # WEG
]

JANELAS = (5, 21, 63)  # dias uteis em 1 semana, 1 mês e 1 trimestre de pregões
HORIZONTE = 21         # horizonte do alvo, em pregões (1 mês)

VOL_FEATURES = [f"log_vol_{w}d" for w in JANELAS] + [f"log_park_{w}d" for w in JANELAS]
TARGET_COL = f"target_log_vol_{HORIZONTE}d_futura"

# =====================================================================
# 3. AQUISIÇÃO DOS DADOS DO YFINANCE
# =====================================================================

def download_b3_data(tickers: list[str], period: str = "10y") -> pd.DataFrame:
    """Baixa dados históricos diários do Yahoo Finance e salva o CSV bruto."""

    all_raw_frames = []

    for ticker in tickers:
        print(f"  Baixando {ticker}...")

        # preços a cada 1 dia
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df["ticker"] = ticker

        df = df[["Date", "ticker", "Open", "High", "Low", "Close", "Volume"]].copy()
        df = df.rename(
            columns={
                "Date": "data",
                "Open": "abertura",
                "High": "maxima",
                "Low": "minima",
                "Close": "fechamento",
                "Volume": "volume",
            }
        )
        all_raw_frames.append(df)

    raw_df = pd.concat(all_raw_frames, ignore_index=True)
    raw_df = raw_df.sort_values(by=["ticker", "data"])
    raw_df.to_csv(RAW_CSV_PATH, index=False)
    return raw_df

def load_raw_data() -> pd.DataFrame:
    """Carrega o CSV bruto versionado (dataset oficial do projeto)."""

    raw_df = pd.read_csv(RAW_CSV_PATH, parse_dates=["data"])
    print(f"  {len(raw_df)} linhas, de {raw_df['data'].min().date()} "
          f"a {raw_df['data'].max().date()}")
    return raw_df

# =====================================================================
# 4. ENGENHARIA DE FEATURES
# =====================================================================

def build_features(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula features de volatilidade (X) e alvo (y) por ticker, gerando também o resumo recente para a API."""
    print("Features de volatilidade e alvo...")
    feature_frames = []
    summary_rows = []

    for ticker, group in raw_df.groupby("ticker"):
        df_t = group.sort_values("data").copy()

        # 1. Bases de cálculo (retornos e variância)
        df_t["retorno_diario"] = df_t["fechamento"].pct_change()
        log_ret = np.log(df_t["fechamento"]).diff()
        r2 = log_ret ** 2
        park_var = np.log(df_t["maxima"] / df_t["minima"]) ** 2 / (4 * np.log(2))

        # 2. Features de volatilidade histórica (X) em múltiplas janelas
        for w in JANELAS:
            df_t[f"log_vol_{w}d"] = 0.5 * np.log(r2.rolling(w).mean()) # é log sigma a partir da média de r^2
            df_t[f"log_park_{w}d"] = 0.5 * np.log(park_var.rolling(w).mean())

        # 3. Alvo (y): volatilidade futura dos próximos 21 pregões
        df_t[TARGET_COL] = 0.5 * np.log(r2.rolling(HORIZONTE).mean().shift(-HORIZONTE)) # o alvo cobre de t+1 a t+21, sem incluir o dia atual.

        # 4. Tratamento de infinitos e filtragem para o dataset de treino
        df_t = df_t.replace([np.inf, -np.inf], np.nan)
        df_clean = df_t.dropna(subset=["retorno_diario", TARGET_COL])
        
        cols_treino = [
            "data", "ticker", "abertura", "maxima", "minima", 
            "fechamento", "volume", "retorno_diario", *VOL_FEATURES, TARGET_COL
        ]
        feature_frames.append(df_clean[cols_treino])

        # 5. Extração do estado mais recente (último pregão) para a API
        ultimo = df_t.dropna(subset=VOL_FEATURES).iloc[-1]
        summary_rows.append({
            "ticker": ticker,
            "data_ultimo_registro": str(pd.Timestamp(ultimo["data"]).date()),
            "ultimo_preco": round(float(ultimo["fechamento"]), 4),
            "media_historica_diaria": float(df_t["retorno_diario"].mean()),
            "desvio_padrao_historico_diario": float(df_t["retorno_diario"].std()),
            **{c: float(ultimo[c]) for c in VOL_FEATURES},
        })

    features_df = pd.concat(feature_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    features_df.to_csv(FEATURES_CSV_PATH, index=False)
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)

    print(f" Features salvas em: {FEATURES_CSV_PATH.name} ({len(features_df)} linhas)")
    print(f" Resumo salvo em: {SUMMARY_CSV_PATH.name}")

    return features_df, summary_df

# =====================================================================
# MAIN
# =====================================================================

def main():
    # Garante que a pasta existe
    DATA_DIR.mkdir(exist_ok=True)

    # Passo 1: Obtenção dos dados brutos dos ultimos 10 anos
    if not RAW_CSV_PATH.exists():
        raw_df = download_b3_data(TICKERS, period="10y")
    else:
        raw_df = load_raw_data()

    # Passo 2: Engenharia de features
    _, summary_df = build_features(raw_df)

    # Passo 3: Exibição amigável do estado mais recente dos ativos
    print("\n[Resumo] Estado mais recente de cada ativo (referência para a API):")
    resumo = summary_df[["ticker", "data_ultimo_registro", "ultimo_preco", "log_vol_21d"]].copy()
    resumo["vol_diaria_21d"] = np.exp(resumo["log_vol_21d"]).map("{:.2%}".format)
    print(resumo.drop(columns="log_vol_21d").to_string(index=False))
    
    print("\nProcesso concluído com sucesso!")

if __name__ == "__main__":
    main()