"""
Fase 3: Ciclo de Vida e Inicialização Otimizada (Lifespan)
- Gerencia o ciclo de vida (Lifespan) para carregar o modelo de ML na RAM uma única vez.
- Inclui os roteadores da API.

Ponto de entrada da API.

O modelo é carregado uma única vez, quando o servidor sobe (lifespan).
As rotas só usam o modelo que já está na memória.

Rodar a partir da raiz do projeto:
    uvicorn app.main:app --reload
Documentação interativa: http://127.0.0.1:8000/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routers import router
from app.services import modelo

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Antes do yield: roda uma vez, na inicialização
    logger.info("Carregando modelo de %s ...", modelo.caminho)
    try:
        modelo.carregar()
        logger.info("Modelo carregado. Tickers: %s", ", ".join(modelo.tickers))
    except Exception:
        # A API sobe mesmo assim: /health mostra modelo_carregado=false e /predict responde 503
        logger.exception("Falha ao carregar o modelo")

    yield  # a aplicação atende às requisições aqui

    # Depois do yield: roda uma vez, no desligamento
    logger.info("Encerrando a API.")


app = FastAPI(
    title="B3 Stock Predictor API",
    description=(
        "Previsão da volatilidade do próximo mês para cinco ações da B3 "
        "(PETR4, VALE3, ITUB4, BBDC4, WEGE3), com intervalo de 90%, "
        "faixa de preço provável e VaR de 95%."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)