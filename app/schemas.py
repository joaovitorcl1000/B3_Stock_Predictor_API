"""
Fase 4: Modelagem e Validação Estrita de Dados (Pydantic)
- Esquemas de Entrada e Saída: Criar classes herdadas de BaseModel em módulo separado (schemas.py), 
definindo explicitamente o contrato de dados que a API aceita e o contrato que ela devolve.

- Uso de Tipagem e Restrições (Field): Configurar tipos estáticos adequados e regras mínimas de negócio nos atributos
 (ex.: faixas de valores com ge, le, gt, limites de tamanho de texto e descrições).

- Prevenção de Dados Inválidos: Garantir que o envio de tipos incorretos (ex.: texto em campos numéricos ou valores fora dos limites) 
seja interceptado automaticamente pelo Pydantic, gerando respostas com status HTTP 422 Unprocessable Entity 
antes de repassar os dados para o modelo preditivo.

Contratos de dados da API (Pydantic).

Definem o que a API aceita e o que devolve. O FastAPI usa essas classes para
validar as requisições (respondendo 422 quando algo está fora das regras) e
para gerar a documentação em /docs.
"""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

# =====================================================================
# ENTRADA
# =====================================================================

class FeaturesVolatilidade(BaseModel):
    """
    Log da volatilidade diária passada, em três janelas.
    Limites de -9 a 0 correspondem a volatilidades entre ~0,01% e 100% ao dia.
    """
    model_config = ConfigDict(extra="forbid")

    log_vol_5d: float = Field(ge=-9, le=0, description="Log da vol. realizada (fechamento), últimos 5 pregões")
    log_vol_21d: float = Field(ge=-9, le=0, description="Log da vol. realizada (fechamento), últimos 21 pregões")
    log_vol_63d: float = Field(ge=-9, le=0, description="Log da vol. realizada (fechamento), últimos 63 pregões")
    log_park_5d: float = Field(ge=-9, le=0, description="Log da vol. de Parkinson (máx./mín.), últimos 5 pregões")
    log_park_21d: float = Field(ge=-9, le=0, description="Log da vol. de Parkinson (máx./mín.), últimos 21 pregões")
    log_park_63d: float = Field(ge=-9, le=0, description="Log da vol. de Parkinson (máx./mín.), últimos 63 pregões")

class PredictRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"ticker": "PETR4.SA", "valor_investido": 10000}]},
    )

    ticker: str = Field(
        min_length=7,
        max_length=9,
        pattern=r"^[A-Z]{4}\d{1,2}\.SA$",
        description="Código da ação na B3 com sufixo .SA (ex.: PETR4.SA)",
    )
    valor_investido: float = Field(
        default=10000,
        gt=0,
        le=1e9,
        description="Valor aplicado na ação, em reais (usado no VaR)",
    )
    features: FeaturesVolatilidade | None = Field(
        default=None,
        description="Opcional. Se omitido, usa as features do último pregão disponível",
    )

# =====================================================================
# SAÍDA
# =====================================================================

class Intervalo(BaseModel):
    inferior: float
    superior: float


class PredictResponse(BaseModel):
    ticker: str
    data_referencia: str = Field(description="Data do último pregão usado como referência")
    origem_features: Literal["ultimo_pregao", "usuario"]
    volatilidade_diaria: float = Field(ge=0, description="Volatilidade diária prevista para o próximo mês")
    volatilidade_mensal: float = Field(ge=0, description="Volatilidade prevista para o mês (diária x raiz de 21)")
    intervalo_vol_mensal_90: Intervalo = Field(description="Intervalo de 90% da volatilidade mensal")
    preco_referencia: float = Field(gt=0, description="Último preço de fechamento conhecido")
    faixa_preco_90: Intervalo = Field(description="Faixa de preço provável no próximo mês (90%)")
    var_95_mensal: float = Field(ge=0, description="Perda que só deve ser superada em 5% dos meses, em reais")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    modelo_carregado: bool
    tickers: list[str]
    data_ultimo_pregao: str | None = None