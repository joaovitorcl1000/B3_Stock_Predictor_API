"""
Regressor MLP em PyTorch compatível com a API do scikit-learn.
Aqui fica a classe que aprende a volatilidade.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin


class StockRegressor(BaseEstimator, RegressorMixin):
    """
    MLP com ativação Tanh treinada por mínimos quadrados.

    Funcao perda: L = mean( w_i * (y_pred_i - y_i)^2 ). Se forem passadas variâncias
    no fit, w_i é proporcional a 1/sigma_i² (um chi², como em física, onde cada
    ponto pesa conforme seu erro experimental). Sem variâncias, todos os pesos
    são iguais e a perda é o MSE comum.

    No modelo de volatilidade usamos o MSE: o alvo é log(sigma), e o erro de
    estimação de log(sigma) é aproximadamente o mesmo em qualquer regime,
    então o chi² se reduz ao MSE.
    """

    def __init__(self, hidden_layer_sizes=(8, 4), lr=1e-3, epochs=1000,
                 weight_decay=1e-4, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.random_state = random_state

    def _build_model(self, input_dim):
        layers = []
        prev_dim = input_dim
        for hidden_dim in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    @staticmethod
    def _to_numpy(a):
        return a.to_numpy() if hasattr(a, "to_numpy") else np.asarray(a)

    def fit(self, X, y, variances=None):
        torch.manual_seed(self.random_state)  # reprodutibilidade

        X_t = torch.tensor(self._to_numpy(X), dtype=torch.float32)

        # Padronização do alvo (desfeita no predict)
        y_np = self._to_numpy(y).astype(float).reshape(-1)
        self.y_mean_ = float(y_np.mean())
        self.y_std_ = float(y_np.std())
        y_t = torch.tensor((y_np - self.y_mean_) / self.y_std_, dtype=torch.float32)

        # Pesos da perda: constantes durante o treino, calculados uma vez só
        if variances is None:
            weights = torch.ones_like(y_t)
        else:
            var_t = torch.tensor(self._to_numpy(variances), dtype=torch.float32).reshape(-1)
            weights = 1.0 / (var_t + 1e-8)
            weights = weights / weights.mean()  # mantém a perda na escala do MSE

        self.model_ = self._build_model(X_t.shape[1])
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)

        self.model_.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            y_pred = self.model_(X_t).squeeze(-1)
            loss = torch.mean(weights * (y_pred - y_t) ** 2)
            loss.backward()
            optimizer.step()

        self.model_.eval()
        return self

    def predict(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("O modelo ainda não foi treinado.")
        X_t = torch.tensor(self._to_numpy(X), dtype=torch.float32)
        with torch.no_grad():
            pred = self.model_(X_t).squeeze(-1).numpy()
        return pred * self.y_std_ + self.y_mean_