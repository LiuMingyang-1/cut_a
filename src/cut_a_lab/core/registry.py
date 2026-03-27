"""Static registries for methods and recipes."""

from __future__ import annotations

from functools import lru_cache

from cut_a_lab.methods.base import BaseMethod
from cut_a_lab.methods.delta_entropy.loader import DeltaEntropyMethod
from cut_a_lab.methods.entropy.loader import EntropyMethod
from cut_a_lab.methods.icr.loader import ICRMethod
from cut_a_lab.recipes.base import RecipeSpec
from cut_a_lab.recipes.cut_a_default import CUT_A_DEFAULT_RECIPE


@lru_cache(maxsize=1)
def method_registry() -> dict[str, BaseMethod]:
    """Return the method registry."""
    return {
        "icr": ICRMethod(),
        "entropy": EntropyMethod(),
        "delta_entropy": DeltaEntropyMethod(),
    }


@lru_cache(maxsize=1)
def recipe_registry() -> dict[str, RecipeSpec]:
    """Return the recipe registry."""
    return {CUT_A_DEFAULT_RECIPE.name: CUT_A_DEFAULT_RECIPE}


def list_methods() -> list[str]:
    """Return sorted method names."""
    return sorted(method_registry())


def list_recipes() -> list[str]:
    """Return sorted recipe names."""
    return sorted(recipe_registry())


def get_method(name: str) -> BaseMethod:
    """Fetch one method by name."""
    registry = method_registry()
    if name not in registry:
        raise KeyError(f"Unknown method {name!r}. Available methods: {', '.join(sorted(registry))}")
    return registry[name]


def get_recipe(name: str) -> RecipeSpec:
    """Fetch one recipe by name."""
    registry = recipe_registry()
    if name not in registry:
        raise KeyError(f"Unknown recipe {name!r}. Available recipes: {', '.join(sorted(registry))}")
    return registry[name]
