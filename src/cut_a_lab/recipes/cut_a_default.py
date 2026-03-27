"""Default Cut A recipe for the first standalone migration."""

from __future__ import annotations

from cut_a_lab.recipes.base import FeatureSetSpec, RecipeSpec


CUT_A_DEFAULT_RECIPE = RecipeSpec(
    name="cut_a_default",
    description=(
        "Standalone Cut A ablation recipe. It reproduces the main prepared-vector "
        "feature set comparisons using independent ICR, entropy, and delta entropy methods."
    ),
    feature_sets=(
        FeatureSetSpec(name="icr_only", methods=("icr",)),
        FeatureSetSpec(name="entropy_only", methods=("entropy",)),
        FeatureSetSpec(name="delta_entropy_only", methods=("delta_entropy",)),
        FeatureSetSpec(name="icr_entropy", methods=("icr", "entropy")),
        FeatureSetSpec(name="icr_delta_entropy", methods=("icr", "delta_entropy")),
        FeatureSetSpec(
            name="discrepancy_combined",
            methods=("icr", "entropy"),
            view_name="discrepancy_combined",
        ),
    ),
)
