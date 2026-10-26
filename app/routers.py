"""
# Fase 5: Implementação de Endpoints e Tratamento de Erros (FastAPI & HTTP)
# - APIRouter e Endpoints: Agrupa as rotas de negócio da aplicação em um módulo dedicado.
# - GET /health: Checa integridade operacional, confirmando presença da Rede Neural em RAM (HTTP 200).
# - POST /predict ou /v1/predict: Consome o payload validado, invoca o service e retorna a predição (HTTP 200).
# - Tratamento de Exceções HTTP: Mapeia status codes com segurança (400 Bad Request, 503 Service Unavailable, 500 Internal Server Error).

Rotas da API.

Recebem a requisição já validada pelo Pydantic (dados inválidos voltam com
422 antes de chegar aqui), chamam a camada de serviço e traduzem as exceções
dela em status HTTP.
"""

import logging

from fastapi import APIRouter, HTTPException, status

from app.schemas import HealthResponse, PredictRequest, PredictResponse
from app.services import ErroDePredicao, ModeloNaoCarregado, TickerNaoSuportado, modelo

logger = logging.getLogger("uvicorn.error")

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Monitoramento"])
def health():
    """Informa se o serviço está no ar e se o modelo está carregado na memória."""
    return HealthResponse(
        status="ok",
        modelo_carregado=modelo.carregado,
        tickers=modelo.tickers,
        data_ultimo_pregao=modelo.data_ultimo_pregao,
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Previsão"],
    responses={
        400: {"description": "Ticker com formato válido, mas não suportado pelo modelo"},
        422: {"description": "Dados inválidos (tipo errado, fora dos limites ou campo desconhecido)"},
        500: {"description": "Erro interno ao calcular a previsão"},
        503: {"description": "Modelo não carregado"},
    },
)
def predict(req: PredictRequest):
    """Prevê a volatilidade do próximo mês e o risco em reais para uma ação."""
    try:
        return modelo.prever(req)
    except TickerNaoSuportado as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ModeloNaoCarregado as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except ErroDePredicao as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception:
        # Qualquer erro inesperado: registra no log e responde 500, sem derrubar a API
        logger.exception("Erro inesperado no /predict")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno ao calcular a previsão.",
        )