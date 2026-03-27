"""Lazy torch model builders."""

from __future__ import annotations

from typing import Any, Callable


def require_torch() -> Any:
    """Import torch lazily so documentation commands still work without it."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency `torch`. Install project dependencies before training.") from exc
    return torch


def _make_baseline_mlp_class():
    torch = require_torch()
    nn = torch.nn
    functional = torch.nn.functional

    class BaselineMLP(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.bn1 = nn.BatchNorm1d(64)
            self.dropout1 = nn.Dropout(0.3)
            self.fc2 = nn.Linear(64, 32)
            self.bn2 = nn.BatchNorm1d(32)
            self.dropout2 = nn.Dropout(0.3)
            self.fc3 = nn.Linear(32, 1)
            self._init_weights()

        def _init_weights(self) -> None:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, a=0.01, nonlinearity="leaky_relu")
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
                elif isinstance(module, nn.BatchNorm1d):
                    nn.init.constant_(module.weight, 1.0)
                    nn.init.constant_(module.bias, 0.0)

        def forward(self, x):
            out = functional.leaky_relu(self.bn1(self.fc1(x)), 0.01)
            out = self.dropout1(out)
            out = functional.leaky_relu(self.bn2(self.fc2(out)), 0.01)
            out = self.dropout2(out)
            return torch.sigmoid(self.fc3(out))

    return BaselineMLP


def build_torch_model_factories() -> dict[str, Callable[[int], Any]]:
    """Return torch model factories used by the default recipe."""
    baseline_mlp = _make_baseline_mlp_class()
    return {"baseline_mlp": lambda input_dim: baseline_mlp(input_dim=input_dim)}
