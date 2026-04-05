"""Default Cut A recipe for the first standalone migration."""

from __future__ import annotations

from cut_a_lab.recipes.base import FeatureSetSpec, RecipeSpec


CUT_A_DEFAULT_RECIPE = RecipeSpec(
    name="cut_a_default",
    description=(
        "Standalone Cut A ablation recipe. Compares ICR and entropy methods independently "
        "and in combination, using a torch MLP classifier."
    ),
    feature_sets=(
        FeatureSetSpec(name="icr_only", methods=("icr",)),
        FeatureSetSpec(name="entropy_only", methods=("entropy",)),
        FeatureSetSpec(name="icr_entropy", methods=("icr", "entropy")),
    ),
)
