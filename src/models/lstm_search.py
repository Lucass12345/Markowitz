"""Validation-only selection; the caller evaluates the winner on untouched test data."""
from itertools import product
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.models.return_model_lstm import ReturnLSTM


def default_candidates():
    return [dict(hidden_sizes=list(sizes), activation=activation, lr=lr, betas=betas,
                 eps=eps, weight_decay=decay)
            for sizes, activation, (lr, betas, eps, decay) in product(
                [(64, 64), (120, 64)], ["relu", "tanh", "gelu"],
                [(1e-3, (0.9, 0.999), 1e-8, 0.0),
                 (3e-4, (0.9, 0.99), 1e-7, 1e-4)])]


def fit_candidates(X_train, y_train, X_val, y_val, candidates, epochs=50,
                   batch_size=32, dropout=0.2, patience=8, seed=42, device="cpu"):
    if not candidates or epochs < 1 or patience < 1:
        raise ValueError("Candidates, epochs and patience must be positive")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    loader = DataLoader(TensorDataset(torch.as_tensor(X_train, dtype=torch.float32),
                                     torch.as_tensor(y_train, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=False)
    xv = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_val, dtype=torch.float32, device=device)
    best = None
    records = []
    for config in candidates:
        torch.manual_seed(seed)
        model = ReturnLSTM(input_size=X_train.shape[-1], dropout=dropout,
                           hidden_sizes=config["hidden_sizes"], activation=config["activation"], feature_indices=config.get("feature_indices")).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                      betas=tuple(config.get("betas", (0.9, 0.999))),
                                      eps=config.get("eps", 1e-8), weight_decay=config.get("weight_decay", 0))
        criterion = nn.MSELoss()
        best_loss, state, stale = np.inf, None, 0
        train_losses, val_losses = [], []
        for _ in range(epochs):
            model.train()
            total, count = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += loss.item() * len(xb)
                count += len(xb)
            model.eval()
            with torch.no_grad():
                value = criterion(model(xv), yv).item()
            train_losses.append(total / count)
            val_losses.append(value)
            if np.isfinite(value) and value < best_loss:
                best_loss, stale = value, 0
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= patience:
                break
        records.append({**config, "val_mse_scaled": best_loss, "epochs_trained": len(train_losses)})
        if state is not None and (best is None or best_loss < best["best_val"]):
            model.load_state_dict(state)
            best = dict(model=model, config=config, best_val=best_loss,
                        train_losses=train_losses, val_losses=val_losses)
    if best is None:
        raise ValueError("No candidate produced a finite validation score")
    best["candidates"] = records
    return best


def refit_deployment(X, y, config, epochs, batch_size=32, dropout=.2, seed=42, device="cpu"):
    """After evaluation, train the selected specification on all labeled history.

    Fixed epochs from validation early stopping; these weights are never scored on
    the holdout. Their scalers and checkpoint are for future inference only.
    """
    x_mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
    x_std = X.reshape(-1, X.shape[-1]).std(axis=0)
    x_std = np.where(x_std < 1e-8, 1., x_std)
    y_mean, y_std = float(y.mean()), max(float(y.std()), 1e-8)
    xs, ys = (X - x_mean) / x_std, (y - y_mean) / y_std
    torch.manual_seed(seed)
    model = ReturnLSTM(input_size=X.shape[-1], hidden_sizes=config["hidden_sizes"],
                       activation=config["activation"], dropout=dropout,
                       feature_indices=config.get("feature_indices")).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                  betas=tuple(config.get("betas", (.9, .999))),
                                  eps=config.get("eps", 1e-8), weight_decay=config.get("weight_decay", 0))
    loader = DataLoader(TensorDataset(torch.tensor(xs), torch.tensor(ys)),
                        batch_size=batch_size, shuffle=False)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(xb.to(device)), yb.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite deployment refit loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
    model.eval()
    return model, x_mean, x_std, y_mean, y_std
