# B3 Stock Predictor API

API REST que estima o **risco do próximo mês** para cinco ações de alta liquidez da B3. Para cada ação, ela prevê a volatilidade dos próximos 21 pregões e a traduz em números úteis para quem investe: a faixa de preço provável e a perda máxima esperada (VaR) para um valor investido.

> Projeto acadêmico (Módulo 2 – SCTEC). Não constitui recomendação de investimento.

---

## 1. Resumo e problema de negócio

Quem investe numa ação precisa saber, antes de tudo, **quanto ela pode oscilar**. Essa informação é usada para dimensionar posições, definir limites de perda e comparar ativos.

Prever **para onde** o preço vai (o retorno) é extremamente difícil: em testes iniciais deste projeto, nenhum modelo superou a simples média histórica. Já a **volatilidade** é previsível, porque períodos agitados tendem a ser seguidos por períodos agitados. Por isso, uma rede neural prevê a volatilidade do próximo mês e, a partir dela, a API informa ao usuário, para uma das ações analisadas e um valor investido, a faixa de preço provável e a perda máxima esperada (VaR).

### Modelo Matemático

O yfinance nos fornece o preço de fechamento da ação no período $t$, denotado por $P(t)$. Definimos o retorno logarítmico diário como:

$$R(t) := \ln\left(\frac{P(t)}{P(t-1)}\right).$$

Como a direção do preço é ruidosa, o modelo é treinado para prever a **volatilidade**, que é a raiz quadrada da variância. A variância diária realizada para um horizonte de $N$ pregões (ex: $N=21$) é a média dos quadrados dos retornos nesse período:

$$\sigma^2_{\text{fut}}(t) = \frac{1}{N} \sum_{i=1}^{N} R(t+i)^2.$$
No código, assumimos a média diária dos retornos como próxima de zero. 

A volatilidade de Parkinson utiliza as cotações máxima e mínima atingidas durante o pregão, fornecendo um estimador mais eficiente para a variância do que o baseado apenas nos preços de fechamento. Sejam $H(t)$ o preço máximo (High) e $L(t)$ o preço mínimo (Low) da ação no dia $t$. A variância de Parkinson diária é definida matematicamente como:

$$\sigma_P^2(t) = \frac{\left[ \ln\left(\frac{H(t)}{L(t)}\right) \right]^2}{4 \ln(2)}.$$

O alvo $y$ da rede neural é o logaritmo da volatilidade diária dos **próximos** $N = 21$ pregões:

$$y(t) = \frac{1}{2} \ln \sigma^2_{\text{fut}}(t) = \ln \sigma_{\text{fut}}(t).$$

As features $\vec{x}$ usam apenas o **passado**: o ticker do ativo e as volatilidades dos últimos $w$ pregões, com $w \in \{5, 21, 63\}$ (semana, mês e trimestre). Para cada janela, calculamos duas estimativas: uma pelos retornos de fechamento e outra pela variância de Parkinson, que usa as máximas e mínimas do dia:

$$\sigma^2_w(t) = \frac{1}{w} \sum_{i=0}^{w-1} R(t-i)^2, \qquad \sigma^2_{P,w}(t) = \frac{1}{w} \sum_{i=0}^{w-1} \sigma^2_P(t-i).$$

Assim como o alvo, todas as volatilidades entram na rede em escala logarítmica:

$$\vec{x}(t) = \Big( \textrm{ticker},\ \ln \sigma_5,\ \ln \sigma_{21},\ \ln \sigma_{63},\ \ln \sigma_{P,5},\ \ln \sigma_{P,21},\ \ln \sigma_{P,63} \Big).$$

Seja $\hat{y}$ a previsão da rede neural e $P(t)$ o último preço conhecido. A API converte a saída do modelo para medidas de risco:

  - **Volatilidade Prevista:** Revertemos o logaritmo para obter a volatilidade diária esperada:
  $$ \hat{\sigma} = \exp(\hat{y}) $$
  
  - **Intervalo da Volatilidade (90\%):** Utilizando o quantil de erro absoluto empírico $k$ (extraído do dataset de validação), o intervalo de confiança para a volatilidade é:

    $$\ln|y - \hat{y}| \leq k \implies \sigma \in \left[ \hat{\sigma}\exp(-k), \; \hat{\sigma}\exp(k) \right], \quad \textrm{em 90\% das vezes}$$

    Em 90% das vezes, o erro que a minha rede neural comete (para mais ou para menos) não ultrapassa o valor de $k$.

  - **Faixa de Preço (90\%):** Projetando os preços para o fim do mês ($N=21$), escalamos a volatilidade diária para o período ($\hat{\sigma}_N = \hat{\sigma}\sqrt{N}$) e a média histórica diária ($\mu_N = \mu N$, no entanto tomamos $\mu \approx 0$). Usando o Z-score de $1.645$ para 90\% de confiança em uma distribuição log-normal:

  $$ P_{\text{min, max}} = P(t) \exp\left( \mu_N \pm 1.645 \cdot \hat{\sigma}\sqrt{N} \right) $$

  - **VaR de 95\% (Value at Risk):** Foca na cauda inferior de 5\% de probabilidade. O pior retorno esperado em 21 dias é:
    $$ R_{\text{pior}} = \mu_N - 1.645 \cdot \hat{\sigma}\sqrt{N} $$
    Para um capital investido $C_0$, a perda máxima financeira (VaR), que só deve ser superada em 1 a cada 20 meses, é:
    $$ \text{VaR} = C_0 \left( 1 - \exp(R_{\text{pior}}) \right) $$

Ações cobertas: `PETR4.SA` (Petrobras), `VALE3.SA` (Vale), `ITUB4.SA` (Itaú Unibanco), `BBDC4.SA` (Bradesco) e `WEGE3.SA` (WEG).

---

## 2. Origem do modelo e dos dados

Projeto próprio, desenvolvido do zero (**Opção B** do enunciado).

- **Fonte:** Yahoo Finance, via biblioteca `yfinance`. Dados públicos, com preços ajustados por proventos e desdobramentos.
- **Período:** 2016-10-20 a 2026-10-20 (10 anos, 2.491 pregões por ação, 12.455 linhas).
- **Por que essa base:** são as cinco ações de maior peso e liquidez da B3, com histórico longo e sem lacunas, o que dá dados suficientes para treinar e avaliar fora da amostra.
- **Dataset oficial:** `data/b3_stocks_raw.csv`, versionado no repositório. Os demais arquivos são gerados a partir dele, então o projeto roda sem acesso à internet.

### Features e alvo (`scripts/data.py`)

| Coluna | Descrição |
|---|---|
| `log_vol_5d`, `log_vol_21d`, `log_vol_63d` | Log da volatilidade realizada (retornos de fechamento) nas últimas 5, 21 e 63 sessões |
| `log_park_5d`, `log_park_21d`, `log_park_63d` | Log da volatilidade de Parkinson, que usa a máxima e a mínima do dia e é menos ruidosa |
| `ticker` | Código da ação (variável categórica) |
| `target_log_vol_21d_futura` | **Alvo:** log da volatilidade realizada dos **próximos** 21 pregões |

As três janelas (semana, mês e trimestre) seguem a ideia do modelo HAR (Corsi, 2009): a volatilidade de longo prazo "ancora" a de curto prazo.

O script também gera `data/b3_summary_stats.csv`, com o estado do último pregão de cada ação (preço e features), que a API usa quando o usuário informa só o ticker.

---

## 3. Modelo e treino (`scripts/train_pipeline.py`)

### Divisão dos dados

A divisão é **cronológica**, com as mesmas datas de corte para todas as ações:

| Conjunto | Período | Linhas |
|---|---|---|
| Treino (70%) | 2016-10-21 a 2023-08-29 | 8.535 |
| Validação (15%) | 2023-09-29 a 2025-02-21 | 1.745 |
| Teste (15%) | 2025-03-27 a 2026-09-18 | 1.855 |

Entre os conjuntos há um intervalo de **21 pregões**. Como o alvo de cada dia usa os 21 pregões seguintes, sem esse intervalo o alvo das últimas linhas do treino conteria dias da validação (vazamento de dados).

### Pré-processamento

Tudo dentro de um `Pipeline` do scikit-learn, ajustado só no treino:

- **Imputação de nulos:** `SimpleImputer` (mediana). Os nulos vêm do início das janelas de 63 pregões.
- **Codificação categórica:** `OneHotEncoder` no ticker.
- **Escalonamento:** `StandardScaler` nas features numéricas.

### Rede neural

MLP em PyTorch (`app/ml/regressor.py`), compatível com a API do scikit-learn:

- ativação **Tanh**, que é suave e não tem neurônios "mortos";
- alvo padronizado internamente, para que a regularização não distorça o nível da previsão;
- otimizador Adam, 5.000 épocas, perda **MSE na escala log**. Na escala log, o erro de estimação da volatilidade é aproximadamente o mesmo em qualquer regime, então o MSE equivale a um χ² com incertezas iguais.

A arquitetura e a regularização foram escolhidas **na validação**, entre 4 arquiteturas e 4 valores de `weight_decay`. O teste não participou da escolha. Em todas as arquiteturas o melhor valor foi `weight_decay = 0,03`, no interior da grade, e a vencedora foi a rede com camadas **(32, 16)**.

### Baselines

- **Persistência:** "a volatilidade do próximo mês será igual à do último mês". É a referência natural para volatilidade.
- **Linear (estilo HAR):** regressão linear com as mesmas features, a referência clássica da literatura.

### Métricas no teste

| Modelo | Skill vs. persistência | R² | Erro relativo mediano da volatilidade |
|---|---|---|---|
| **Rede (32, 16)** | **+0,313** | −0,073 | 20,3% |
| Linear (HAR) | +0,301 | −0,092 | 19,9% |
| Persistência | 0 | −0,561 | 23,1% |

- **Skill** = 1 − MSE do modelo / MSE da persistência: a fração do erro da persistência que o modelo elimina. A rede reduz em cerca de **31%** o erro de simplesmente repetir o último mês.
- Rede e HAR ficam praticamente empatados (na validação o HAR é 1% melhor; no teste a rede é 1,7% melhor). A rede alcança a referência da literatura, sem superá-la de forma significativa.
- **R² negativo:** o R² compara cada modelo com a média do próprio período de teste, que nenhum modelo conhece de antemão. O teste (2025–2026) teve pouca variação de volatilidade para explicar, então até modelos úteis ficam abaixo dessa referência. Por isso a métrica principal é o skill.
- **Intervalo de 90%:** calibrado nos resíduos da validação. No teste, cobriu **92,3%** dos casos. A volatilidade real ficou entre a prevista ÷ 1,80 e × 1,80.

Todas as métricas, incluindo a grade completa, estão em `models/metrics.json`.

---

## 4. Tecnologias utilizadas

| Biblioteca | Versão | Uso |
|---|---|---|
| Python | 3.11 | |
| FastAPI | 0.162.0 | API REST |
| Uvicorn | 0.54.0 | Servidor ASGI |
| Pydantic | 2.14.1 | Validação de entrada e saída |
| PyTorch | 2.14.1 | Rede neural |
| scikit-learn | 1.9.1 | Pré-processamento, pipeline e baseline linear |
| joblib | 1.6.0 | Serialização do modelo |
| pandas | 3.1.0 | Manipulação de dados |
| NumPy | 2.4.6 | Cálculos numéricos |
| yfinance | 1.7.0 | Download dos dados |

---

## 5. Instalação e execução

Todos os comandos são executados **na raiz do projeto**.

```bash
git clone https://github.com/joaovitorcl1000/B3_Stock_Predictor_API.git
cd B3_Stock_Predictor_API

python3 -m venv venv
source venv/bin/activate          # no Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> O `torch` padrão do pip inclui suporte a GPU e é um download grande. Para usar só a CPU, instale-o antes:
> `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu`

O repositório já contém os dados e o modelo treinado. Para iniciar a API:

```bash
uvicorn app.main:app --reload
```

Opcionalmente, para regenerar as features e retreinar o modelo:

```bash
python -m scripts.data              # reconstrói as features a partir do CSV bruto versionado
python -m scripts.train_pipeline    # treina, avalia e salva o modelo em models/
```

O `scripts.data` não acessa a internet se `data/b3_stocks_raw.csv` existir. Para baixar uma janela nova do Yahoo Finance, apague esse arquivo antes; os resultados vão mudar.

---

## 6. Documentação da API

Com o servidor rodando, a documentação interativa (Swagger) fica em **http://127.0.0.1:8000/docs**.

O modelo é carregado **uma única vez**, quando o servidor sobe (`lifespan` em `app/main.py`). As requisições usam o modelo que já está na memória, sem ler o disco.

### `GET /health`

Informa se o serviço está no ar e se o modelo está carregado.

```json
{
  "status": "ok",
  "modelo_carregado": true,
  "tickers": ["BBDC4.SA", "ITUB4.SA", "PETR4.SA", "VALE3.SA", "WEGE3.SA"],
  "data_ultimo_pregao": "2026-10-20"
}
```

### `POST /predict`

**Requisição mínima:** só o ticker e o valor investido. A API usa as features do último pregão disponível.

```json
{
  "ticker": "PETR4.SA",
  "valor_investido": 10000
}
```

**Resposta (200):**

```json
{
  "ticker": "PETR4.SA",
  "data_referencia": "2026-10-20",
  "origem_features": "ultimo_pregao",
  "volatilidade_diaria": 0.022806,
  "volatilidade_mensal": 0.104511,
  "intervalo_vol_mensal_90": {"inferior": 0.058001, "superior": 0.188317},
  "preco_referencia": 58.21,
  "faixa_preco_90": {"inferior": 49.02, "superior": 69.13},
  "var_95_mensal": 1579.54
}
```

Leitura: com 90% de confiança, a PETR4 deve terminar o próximo mês entre R$ 49,02 e R$ 69,13; para R$ 10.000 investidos, a perda só deve passar de R$ 1.579,54 em 1 de cada 20 meses.

**Requisição com features próprias**, para simular um cenário (por exemplo, um mês de volatilidade alta):

```json
{
  "ticker": "VALE3.SA",
  "valor_investido": 5000,
  "features": {
    "log_vol_5d": -3.2,
    "log_vol_21d": -3.4,
    "log_vol_63d": -3.8,
    "log_park_5d": -3.3,
    "log_park_21d": -3.5,
    "log_park_63d": -3.9
  }
}
```

As features são o log da volatilidade diária: −3,4 corresponde a e^−3,4 ≈ 3,3% ao dia. Nesse caso, a resposta vem com `"origem_features": "usuario"`.

### Erros

| Status | Quando | Exemplo |
|---|---|---|
| **400** | Ticker com formato válido, mas fora do modelo | `{"ticker": "MGLU3.SA"}` |
| **422** | Dados inválidos, barrados pelo Pydantic antes de chegar ao modelo | `{"ticker": "abc"}`, `{"ticker": "PETR4.SA", "valor_investido": -5}`, texto em campo numérico, campo desconhecido |
| **500** | Erro inesperado no cálculo (registrado no log; a API continua no ar) | |
| **503** | Modelo não carregado na memória | |

Exemplo de resposta 400:

```json
{"detail": "Ticker MGLU3.SA não suportado. Disponíveis: BBDC4.SA, ITUB4.SA, PETR4.SA, VALE3.SA, WEGE3.SA"}
```

### Como o risco é calculado

- volatilidade mensal = volatilidade diária × √21;
- faixa de preço = preço × e^(±1,645 × volatilidade mensal);
- VaR de 95% = valor × (1 − e^(−1,645 × volatilidade mensal)).

As fórmulas supõem retorno médio zero e preço com distribuição lognormal. Na prática, retornos têm caudas mais pesadas que a normal, então o VaR tende a ser um pouco otimista em meses extremos.

---

## 7. Estrutura do projeto

```
├── app/
│   ├── main.py          # ponto de entrada e lifespan (carrega o modelo)
│   ├── routers.py       # rotas /health e /predict, status HTTP
│   ├── schemas.py       # contratos de entrada e saída (Pydantic)
│   ├── services.py      # carregamento do modelo e cálculos (sem HTTP)
│   └── ml/
│       └── regressor.py # rede neural em PyTorch
├── scripts/
│   ├── data.py          # aquisição de dados e features
│   └── train_pipeline.py# treino, avaliação e serialização
├── data/                # dataset bruto, features e resumo por ação
├── models/              # modelo treinado (.joblib) e métricas (.json)
└── requirements.txt
```

A camada de serviço não conhece HTTP: em caso de problema, lança exceções próprias (`TickerNaoSuportado`, `ModeloNaoCarregado`), e as rotas as traduzem em status 400, 503 ou 500.

---

## 8. Organização do desenvolvimento

As tarefas foram organizadas num quadro Kanban no Trello: [B3 Stock Predictor API – Kanban](https://trello.com/b/20n2Qzpn/b3stockpredictorapi).


O projeto seguiu o GitFlow: a `main` recebe só a versão final; a `develop` integra as funcionalidades; cada tarefa teve sua própria branch, integrada por pull request.

| Branch | Conteúdo |
|---|---|
| `feature/configuracao-ambiente` | Dependências e estrutura de pacotes |
| `feature/aquisicao-dados` | Download, features e dataset versionado |
| `fix/requirements` | Versões fixas e padronização de quebras de linha |
| `feature/modelo-regressor` | Rede neural em PyTorch |
| `feature/pipeline-treino` | Treino, seleção, baselines e métricas |
| `fix/data` | Documentação do script de dados |
| `feature/api-schemas` | Contratos Pydantic |
| `feature/api-services` | Camada de serviço |
| `feature/api-endpoints` | Rotas e lifespan |
| `docs/readme` | Este README |

---

## 9. Evidências

- **Vídeo de apresentação:** 
- **Quadro Kanban (Trello):** https://trello.com/b/20n2Qzpn/b3stockpredictorapi