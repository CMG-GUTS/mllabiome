from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class _JointMLP(BaseEstimator):
    def __init__(
        self,
        hidden_layer_sizes=(128, 64),
        dropout=0.1,
        learning_rate=0.001,
        weight_decay=0.0001,
        epochs=100,
        batch_size=32,
        random_state=42,
        device="cpu",
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device

    def _fit(self, X, y, classification):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("JointMLP requires PyTorch.") from exc
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim != 2 or y.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError("JointMLP requires X and y to be aligned two-dimensional arrays.")
        if X.shape[1] == 0 or y.shape[1] == 0 or not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("JointMLP requires nonempty finite features and targets.")
        if classification and not np.isin(y, [0, 1]).all():
            raise ValueError("Multilabel targets must contain only 0 and 1.")
        if not 0 <= float(self.dropout) < 1 or int(self.epochs) < 1 or int(self.batch_size) < 1:
            raise ValueError("Invalid dropout, epochs or batch size.")
        if float(self.learning_rate) <= 0 or float(self.weight_decay) < 0:
            raise ValueError("Invalid optimizer parameters.")
        hidden = tuple(int(v) for v in self.hidden_layer_sizes)
        if any(v < 1 for v in hidden):
            raise ValueError("Every hidden layer must contain at least one neuron.")
        torch.manual_seed(int(self.random_state))
        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = y.shape[1]
        self._classification = classification
        self._device = torch.device(self.device)
        self.x_mean_ = X.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.x_scale_ = X.std(axis=0, dtype=np.float64).astype(np.float32)
        self.x_scale_[self.x_scale_ < 1e-8] = 1.0
        X = (X - self.x_mean_) / self.x_scale_
        self.y_mean_ = np.zeros(y.shape[1], dtype=np.float32) if classification else y.mean(axis=0)
        self.y_scale_ = np.ones(y.shape[1], dtype=np.float32) if classification else y.std(axis=0)
        self.y_scale_[self.y_scale_ < 1e-8] = 1.0
        if not classification:
            y = (y - self.y_mean_) / self.y_scale_
        layers = []
        width = X.shape[1]
        for size in hidden:
            layers.extend((torch.nn.Linear(width, size), torch.nn.ReLU()))
            if self.dropout:
                layers.append(torch.nn.Dropout(float(self.dropout)))
            width = size
        layers.append(torch.nn.Linear(width, y.shape[1]))
        self.model_ = torch.nn.Sequential(*layers).to(self._device)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=float(self.learning_rate), weight_decay=float(self.weight_decay))
        criterion = torch.nn.BCEWithLogitsLoss() if classification else torch.nn.MSELoss()
        X_tensor = torch.from_numpy(np.ascontiguousarray(X)).to(self._device)
        y_tensor = torch.from_numpy(np.ascontiguousarray(y)).to(self._device)
        generator = torch.Generator().manual_seed(int(self.random_state))
        self.model_.train()
        for _ in range(int(self.epochs)):
            order = torch.randperm(len(X_tensor), generator=generator)
            for batch in order.split(int(self.batch_size)):
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model_(X_tensor[batch.to(self._device)]), y_tensor[batch.to(self._device)])
                loss.backward()
                optimizer.step()
        self.model_.eval()
        return self

    def _forward(self, X):
        check_is_fitted(self, "model_")
        import torch
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature dimensions differ from training.")
        if not np.isfinite(X).all():
            raise ValueError("Prediction features must be finite.")
        X = (X - self.x_mean_) / self.x_scale_
        with torch.no_grad():
            return self.model_(torch.from_numpy(np.ascontiguousarray(X)).to(self._device)).cpu().numpy()


class JointMLPClassifier(_JointMLP, ClassifierMixin):
    def fit(self, X, y):
        self._fit(X, y, True)
        self.classes_ = [np.array([0, 1], dtype=int) for _ in range(self.n_outputs_)]
        return self

    def predict_proba(self, X):
        z = self._forward(X)
        return expit(z)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class JointMLPRegressor(_JointMLP, RegressorMixin):
    def fit(self, X, y):
        return self._fit(X, y, False)

    def predict(self, X):
        return self._forward(X) * self.y_scale_ + self.y_mean_
