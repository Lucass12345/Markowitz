import torch.nn as nn


class ReturnLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2,
                 dense_size=32, hidden_sizes=None, activation="relu", feature_indices=None):
        super().__init__()
        activations = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "elu": nn.ELU}
        if activation not in activations:
            raise ValueError(f"Unsupported dense activation: {activation}")
        self.feature_indices = feature_indices
        if feature_indices is not None:
            input_size = len(feature_indices)
        self.hidden_sizes = list(hidden_sizes) if hidden_sizes is not None else [hidden_size] * num_layers
        if not self.hidden_sizes or min(self.hidden_sizes) < 1:
            raise ValueError("LSTM layer sizes must be positive")
        self.variable_stack = hidden_sizes is not None
        if self.variable_stack:
            sizes = [input_size] + self.hidden_sizes
            self.layers = nn.ModuleList([nn.LSTM(sizes[i], sizes[i+1], batch_first=True)
                                         for i in range(len(self.hidden_sizes))])
            self.inter_dropout = nn.Dropout(dropout)
        else:
            # Preserve state keys for historical 64x64 checkpoints.
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.head = nn.Sequential(nn.Linear(self.hidden_sizes[-1], dense_size),
                                  activations[activation](), nn.Dropout(dropout), nn.Linear(dense_size, 1))

    def forward(self, x):
        if self.feature_indices is not None:
            x = x[:, :, self.feature_indices]
        if self.variable_stack:
            for i, layer in enumerate(self.layers):
                x, _ = layer(x)
                if i < len(self.layers) - 1:
                    x = self.inter_dropout(x)
        else:
            x, _ = self.lstm(x)
        return self.head(x[:, -1, :]).squeeze(-1)
