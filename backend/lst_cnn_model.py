"""
Option 2: Deep Learning Neural Network / CNN Model for LST Heatmap Prediction.
Takes multi-band spatial imagery / engineered feature patches as input,
predicting pixel-wise Land Surface Temperature (LST).
Supports PyTorch (if available) and Scikit-Learn MLP Neural Network fallback.
"""

import numpy as np

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    torch = None

from sklearn.neural_network import MLPRegressor


if TORCH_AVAILABLE:
    class SpatialLSTCNN(nn.Module):
        """
        PyTorch Spatial Convolutional Neural Network / Deep MLP for Satellite Patch LST Regressor
        """
        def __init__(self, input_dim=16):
            super(SpatialLSTCNN, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.encoder(x)


class DeepLearningLSTWrapper:
    def __init__(self, input_dim=16, lr=0.003, epochs=25):
        self.input_dim = input_dim
        self.lr = lr
        self.epochs = epochs
        self.use_torch = TORCH_AVAILABLE
        self.y_mean = 0.0
        self.y_std = 1.0

        if self.use_torch:
            self.model = SpatialLSTCNN(input_dim=input_dim)
            self.criterion = nn.MSELoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        else:
            print("PyTorch not found; utilizing Deep Neural Network (MLPRegressor) for Option 2.")
            self.model = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                max_iter=150,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.y_mean = float(np.mean(y_train))
        self.y_std = float(np.std(y_train)) if np.std(y_train) > 0 else 1.0
        y_train_norm = (y_train - self.y_mean) / self.y_std

        if self.use_torch:
            self.model.train()
            X_t = torch.tensor(X_train, dtype=torch.float32)
            y_t = torch.tensor(y_train_norm, dtype=torch.float32).unsqueeze(1)

            dataset = torch.utils.data.TensorDataset(X_t, y_t)
            loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

            for epoch in range(self.epochs):
                for batch_x, batch_y in loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()
        else:
            self.model.fit(X_train, y_train_norm)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.use_torch:
            self.model.eval()
            with torch.no_grad():
                X_t = torch.tensor(X, dtype=torch.float32)
                preds_norm = self.model(X_t).numpy().flatten()
        else:
            preds_norm = self.model.predict(X)

        return preds_norm * self.y_std + self.y_mean
