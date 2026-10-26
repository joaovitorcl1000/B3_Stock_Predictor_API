"""
Fases 2: Módulo responsável por carregar o modelo serializado e executar a inferência matemática, 
desacoplando regras de negócio do protocolo HTTP.

Camada de serviço: carrega o modelo e faz a inferência.

Não sabe nada de HTTP. Em caso de problema, lança exceções próprias
(TickerNaoSuportado, ModeloNaoCarregado, ErroDePredicao), e a camada de
rotas decide qual status HTTP devolver.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from app.schemas import Intervalo, PredictRequest, PredictResponse

CAMINHO_MODELO = Path(__file__).resolve().parent.parent / "models" / "stock_mlp_bundle.joblib"

DIAS_NO_MES = 21  # pregões no horizonte do modelo
Z = 1.645         # quantil da normal: 5% em cada cauda (faixa de 90% e VaR de 95%)

class ModeloNaoCarregado(Exception):
    """O modelo ainda não foi carregado na memória."""


class TickerNaoSuportado(Exception):
    """O ticker tem formato válido, mas não faz parte do modelo."""


class ErroDePredicao(Exception):
    """O modelo devolveu um valor inválido (NaN ou infinito)."""


class ModeloVolatilidade:
    def __init__(self, caminho: Path = CAMINHO_MODELO):
        self.caminho = caminho
        self.bundle = None

    # -----------------------------------------------------------------
    # Carregamento (chamado uma única vez, no lifespan da aplicação)
    # -----------------------------------------------------------------
    def carregar(self):
        self.bundle = joblib.load(self.caminho)

    @property
    def carregado(self) -> bool:
        return self.bundle is not None

    @property
    def tickers(self) -> list[str]:
        return list(self.bundle["tickers"]) if self.carregado else []

    @property
    def data_ultimo_pregao(self) -> str | None:
        if not self.carregado:
            return None
        return max(estado["data"] for estado in self.bundle["estado_atual"].values())

    # -----------------------------------------------------------------
    # Inferência
    # -----------------------------------------------------------------
    def prever(self, req: PredictRequest) -> PredictResponse:
        if not self.carregado:
            raise ModeloNaoCarregado("O modelo não está carregado.")
        if req.ticker not in self.bundle["tickers"]:
            raise TickerNaoSuportado(
                f"Ticker {req.ticker} não suportado. Disponíveis: {', '.join(self.tickers)}"
            )

        estado = self.bundle["estado_atual"][req.ticker]

        # Features enviadas pelo usuário ou, se não vierem, as do último pregão
        if req.features is not None:
            features = req.features.model_dump()
            origem = "usuario"
        else:
            features = {c: estado[c] for c in self.bundle["features"] if c != "ticker"}
            origem = "ultimo_pregao"

        # Uma linha com as colunas na mesma ordem usada no treino
        linha = pd.DataFrame([{"ticker": req.ticker, **features}])[self.bundle["features"]]
        log_vol = float(self.bundle["pipeline"].predict(linha)[0])
        if not np.isfinite(log_vol):
            raise ErroDePredicao("O modelo devolveu um valor inválido.")

        # O modelo prevê log da vol. diária; exp desfaz o log, raiz de 21 leva ao mês
        k = self.bundle["k"]
        vol_diaria = float(np.exp(log_vol))
        vol_mensal = vol_diaria * np.sqrt(DIAS_NO_MES)
        vol_mensal_inf = float(np.exp(log_vol - k) * np.sqrt(DIAS_NO_MES))
        vol_mensal_sup = float(np.exp(log_vol + k) * np.sqrt(DIAS_NO_MES))

        # Risco em reais, supondo retorno médio zero e preço lognormal
        preco = estado["ultimo_preco"]
        faixa_inf = preco * np.exp(-Z * vol_mensal)
        faixa_sup = preco * np.exp(Z * vol_mensal)
        var_95 = req.valor_investido * (1 - np.exp(-Z * vol_mensal))

        return PredictResponse(
            ticker=req.ticker,
            data_referencia=estado["data"],
            origem_features=origem,
            volatilidade_diaria=round(vol_diaria, 6),
            volatilidade_mensal=round(float(vol_mensal), 6),
            intervalo_vol_mensal_90=Intervalo(
                inferior=round(vol_mensal_inf, 6), superior=round(vol_mensal_sup, 6)
            ),
            preco_referencia=round(preco, 2),
            faixa_preco_90=Intervalo(
                inferior=round(float(faixa_inf), 2), superior=round(float(faixa_sup), 2)
            ),
            var_95_mensal=round(float(var_95), 2),
        )


# Instância única usada pela aplicação (carregada no lifespan)
modelo = ModeloVolatilidade()