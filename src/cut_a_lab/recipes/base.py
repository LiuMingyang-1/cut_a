"""Recipe contracts for combining methods into experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSetSpec:
    """A named feature-set combination used during training."""

    name: str
    methods: tuple[str, ...]
    view_name: str = "concat"


@dataclass(frozen=True)
class RecipeSpec:
    """A recipe describing which methods are active."""

    name: str
    description: str
    feature_sets: tuple[FeatureSetSpec, ...]

    @property
    def method_names(self) -> tuple[str, ...]:
        """Return the distinct methods required by the recipe."""
        ordered: list[str] = []
        for feature_set in self.feature_sets:
            for method_name in feature_set.methods:
                if method_name not in ordered:
                    ordered.append(method_name)
        return tuple(ordered)

    def describe(self) -> str:
        """Return a CLI-friendly summary."""
        lines = [f"Recipe: {self.name}", "", self.description, "", "Feature sets:"]
        for feature_set in self.feature_sets:
            view_suffix = "" if feature_set.view_name == "concat" else f" [view={feature_set.view_name}]"
            lines.append(f"- {feature_set.name}: {', '.join(feature_set.methods)}{view_suffix}")
        return "\n".join(lines)
