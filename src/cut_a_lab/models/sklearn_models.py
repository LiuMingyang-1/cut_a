"""Lazy sklearn model builders."""

from __future__ import annotations

from typing import Any, Callable


def require_sklearn() -> Any:
    """Import sklearn lazily so non-training commands still work without it."""
    try:
        import sklearn
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency `scikit-learn`. Install project dependencies before training."
        ) from exc
    return sklearn


def build_sklearn_model_factories(*, seed: int) -> dict[str, Callable[[], Any]]:
    """Return sklearn model factories used by the default recipe."""
    require_sklearn()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "logistic_regression": lambda: Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        solver="liblinear",
                        random_state=seed,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced_subsample",
        ),
    }
