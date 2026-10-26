"""
Fase 1: Preparação e Serialização do Modelo (Pipeline Offline)

- Treinar (ou refinar) o modelo preditivo escolhido, garantindo a divisão entre dados de treino, validação e teste 
e os devidos tratamentos de pré-processamento (imputação de nulos, codificação categórica e escalonamento).
- Serialização do Artefato: Salvar o pipeline ou modelo treinado em um arquivo binário dedicado (ex.: .joblib ou .pkl)
 dentro de uma pasta apropriada do projeto (ex.: models/) ou usar um banco de dados sqlite, como explorado em aula 
 ou ainda uma ferramenta mais apropriada e robusta como o MLflow.
- Relatório de Métricas: Apresentar em documentação ou script auxiliar as métricas alcançadas pelo modelo selecionado 
(Acurácia, F1-Score, ROC-AUC, MSE/R² dependendo do tipo de problema).

Resumo: Treino do modelo de volatilidade (Fase 1). O modelo prevê o log da volatilidade diária dos próximos 21 pregões a partir
da volatilidade passada (janelas de 5, 21 e 63 pregões) e do ticker.

Passos:
1. Divide os dados por data em treino (70%), validação (15%) e teste (15%).
2. Pré-processa: imputação de nulos, one-hot do ticker e escalonamento.
3. Testa várias arquiteturas da rede e escolhe a melhor na validação.
4. Compara com dois baselines: persistência e regressão linear (estilo HAR).
5. Calibra um intervalo de 90% para a previsão.
6. Salva o modelo em models/ e as métricas em models/metrics.json.

Rodar a partir da raiz do projeto:
    python -m scripts.train_pipeline
"""

import json
from itertools import product
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml.regressor import StockRegressor

# =====================================================================
# CONFIGURAÇÃO
# =====================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_CSV = BASE_DIR / "data" / "b3_stocks_features.csv"
SUMMARY_CSV = BASE_DIR / "data" / "b3_summary_stats.csv"
MODELS_DIR = BASE_DIR / "models"

TARGET = "target_log_vol_21d_futura"
FEATURES_NUM = [
    "log_vol_5d", "log_vol_21d", "log_vol_63d",     # vol. pelos retornos de fechamento
    "log_park_5d", "log_park_21d", "log_park_63d",  # vol. de Parkinson (máxima e mínima)
]
FEATURES_CAT = ["ticker"]

"""
O alvo usa os 21 pregões seguintes. Deixamos 21 pregões de intervalo entre treino, validação e teste 
para que nenhum alvo "enxergue" o conjunto seguinte.
"""
GAP = 21

GRID_CAMADAS = [(8, 4), (16, 8), (32, 16), (16, 16, 8)]
GRID_WEIGHT_DECAY = [1e-2, 3e-2, 1e-1, 3e-1]
LR = 1e-3
EPOCHS = 5000

# =====================================================================
# FUNÇÕES
# =====================================================================

def criar_preprocessador():
    """Imputação + escalonamento nas numéricas, one-hot no ticker."""
    numericas = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # NaN no início das janelas de 63 dias
        ("scaler", StandardScaler()),
    ])
    categoricas = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numericas, FEATURES_NUM),
        ("cat", categoricas, FEATURES_CAT),
    ])


def avaliar(y_true, y_pred, mse_persistencia):
    """Métricas de regressão (escala log) e comparação com a persistência."""
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        # fração do erro da persistência que o modelo elimina (> 0 é melhor)
        "skill_vs_persistencia": 1 - mse / mse_persistencia,
        # erro típico na volatilidade em si: |vol_prevista / vol_real - 1|
        "erro_relativo_mediano": float(np.median(np.abs(np.exp(y_pred - y_true) - 1))),
    }

# =====================================================================
# MAIN
# =====================================================================

def main():
    if not FEATURES_CSV.exists():
        raise FileNotFoundError("Dataset não encontrado. Rode antes: python -m scripts.data")
    MODELS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(FEATURES_CSV, parse_dates=["data"])
    df = df.sort_values(["data", "ticker"]).reset_index(drop=True)
    print(f"Dataset: {len(df)} linhas")

    # -----------------------------------------------------------------
    # 1. Divisão por data (mesmos cortes para todos os tickers)
    # -----------------------------------------------------------------
    datas = np.sort(df["data"].unique())
    corte_val = int(len(datas) * 0.70)
    corte_teste = int(len(datas) * 0.85)

    treino = df["data"] < datas[corte_val - GAP]
    validacao = (df["data"] >= datas[corte_val]) & (df["data"] < datas[corte_teste - GAP])
    teste = df["data"] >= datas[corte_teste]

    divisao = {}
    for nome, m in [("treino", treino), ("validacao", validacao), ("teste", teste)]:
        divisao[nome] = {
            "inicio": str(df.loc[m, "data"].min().date()),
            "fim": str(df.loc[m, "data"].max().date()),
            "linhas": int(m.sum()),
        }
        print(f"  {nome:<10} {divisao[nome]['inicio']} a {divisao[nome]['fim']} "
              f"({divisao[nome]['linhas']} linhas)")

    X = df[FEATURES_CAT + FEATURES_NUM]
    y = df[TARGET]
    X_tr, y_tr = X[treino], y[treino]
    X_va, y_va = X[validacao], y[validacao]
    X_te, y_te = X[teste], y[teste]

    # -----------------------------------------------------------------
    # 2. Baselines
    # Persistência: "a volatilidade do próximo mês é igual à do último mês".
    # Linear: regressão com as mesmas features, no estilo do modelo HAR.
    # -----------------------------------------------------------------
    persist_va = X_va["log_vol_21d"].to_numpy()
    persist_te = X_te["log_vol_21d"].to_numpy()
    mse_persist_va = float(mean_squared_error(y_va, persist_va))
    mse_persist_te = float(mean_squared_error(y_te, persist_te))

    linear = Pipeline([("preprocessor", criar_preprocessador()), ("regressor", LinearRegression())])
    linear.fit(X_tr, y_tr)
    mse_linear_va = float(mean_squared_error(y_va, linear.predict(X_va)))

    print(f"\nMSE validação  persistência={mse_persist_va:.4f}  linear={mse_linear_va:.4f}")

    # -----------------------------------------------------------------
    # 3. Escolha da rede na validação
    # Cada combinação treina só no treino e é avaliada na validação.
    # O teste fica guardado para a avaliação final.
    # -----------------------------------------------------------------
    print("\nBusca de hiperparâmetros:")
    grid = []
    melhor_mse, modelo = None, None

    for camadas, wd in product(GRID_CAMADAS, GRID_WEIGHT_DECAY):
        pipe = Pipeline([
            ("preprocessor", criar_preprocessador()),
            ("regressor", StockRegressor(hidden_layer_sizes=camadas, lr=LR,
                                         epochs=EPOCHS, weight_decay=wd)),
        ])
        pipe.fit(X_tr, y_tr)
        mse_va = float(mean_squared_error(y_va, pipe.predict(X_va)))
        grid.append({"camadas": list(camadas), "weight_decay": wd, "mse_validacao": mse_va})
        print(f"  {str(camadas):<12} wd={wd:<6g} MSE val={mse_va:.4f} "
              f"(x persistência: {mse_va / mse_persist_va:.3f})")

        if melhor_mse is None or mse_va < melhor_mse:
            melhor_mse, modelo = mse_va, pipe

    rede = modelo.named_steps["regressor"]
    print(f"Escolhido: {rede.hidden_layer_sizes}, weight_decay={rede.weight_decay:g}")

    # -----------------------------------------------------------------
    # 4. Intervalo de 90%
    # -----------------------------------------------------------------
    pred_va = modelo.predict(X_va)
    pred_te = modelo.predict(X_te)

    k = float(np.quantile(np.abs(y_va - pred_va), 0.90))
    cobertura = float(np.mean(np.abs(y_te - pred_te) <= k))

    # -----------------------------------------------------------------
    # 5. Métricas finais (teste avaliado uma única vez)
    # -----------------------------------------------------------------
    metricas = {
        "alvo": "log da volatilidade diária dos próximos 21 pregões",
        "divisao": divisao,
        "hiperparametros": rede.get_params(),
        "grid_validacao": grid,
        "teste": {
            "rede": avaliar(y_te, pred_te, mse_persist_te),
            "persistencia": avaliar(y_te, persist_te, mse_persist_te),
            "linear_har": avaliar(y_te, linear.predict(X_te), mse_persist_te),
        },
        "intervalo_90": {"k": k, "fator": float(np.exp(k)), "cobertura_teste": cobertura},
    }

    print("\nTeste:")
    for nome, m in metricas["teste"].items():
        print(f"  {nome:<13} R²={m['r2']:+.3f}  skill={m['skill_vs_persistencia']:+.3f}  "
              f"erro rel.={m['erro_relativo_mediano']:.1%}")
    print(f"  Intervalo 90%: fator x{np.exp(k):.2f}, cobertura no teste = {cobertura:.1%}")

    # -----------------------------------------------------------------
    # 6. Estado atual de cada ativo (a API usa como referência)
    # -----------------------------------------------------------------
    resumo = pd.read_csv(SUMMARY_CSV)
    estado_atual = {
        linha["ticker"]: {
            "data": linha["data_ultimo_registro"],
            "ultimo_preco": float(linha["ultimo_preco"]),
            **{c: float(linha[c]) for c in FEATURES_NUM},
        }
        for _, linha in resumo.iterrows()
    }

    # -----------------------------------------------------------------
    # 7. Salva tudo num único arquivo, carregado uma vez pela API
    # -----------------------------------------------------------------
    bundle = {
        "pipeline": modelo,
        "k": k,
        "features": FEATURES_CAT + FEATURES_NUM,
        "tickers": sorted(df["ticker"].unique()),
        "estado_atual": estado_atual,
        "metricas": metricas,
    }
    joblib.dump(bundle, MODELS_DIR / "stock_mlp_bundle.joblib")
    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=4, ensure_ascii=False)

    print("\nModelo salvo em models/stock_mlp_bundle.joblib")
    print("Métricas salvas em models/metrics.json")


if __name__ == "__main__":
    main()