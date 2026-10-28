"""
Exemplo de como outra aplicação consome a API.

Com a API rodando (uvicorn app.main:app), execute a partir da raiz do projeto:
    python scripts/exemplo_cliente.py
"""

import requests

url = "http://127.0.0.1:8000/predict"
payload = {
    "ticker": "PETR4.SA",
    "valor_investido": 15000,
}

# Faz a requisição POST
try:
    resposta = requests.post(url, json=payload, timeout=10)
except requests.ConnectionError:
    raise SystemExit("API fora do ar. Inicie com: uvicorn app.main:app")

# 400 ou 422: a API devolve o motivo no campo "detail"
if resposta.status_code != 200:
    raise SystemExit(f"Erro {resposta.status_code}: {resposta.json()['detail']}")

# Extrai e imprime o JSON da resposta
dados = resposta.json()
faixa = dados["faixa_preco_90"]

print(f"Ação: {dados['ticker']} (dados até {dados['data_referencia']})")
print(f"Volatilidade prevista para o próximo mês: {dados['volatilidade_mensal']:.1%}")
print(f"Preço atual: R$ {dados['preco_referencia']:.2f}")
print(f"Faixa de preço provável (90%): R$ {faixa['inferior']:.2f} a R$ {faixa['superior']:.2f}")
print(f"Perda máxima estimada (VaR 95%) para R$ {payload['valor_investido']:,.2f}: "
      f"R$ {dados['var_95_mensal']:,.2f}")
