"""Base classes for feature methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from cut_a_lab.core.contracts import FeatureBlock, MethodInputContract


class BaseMethod(ABC):
    """Abstract interface implemented by all feature methods."""

    name: str

    @abstractmethod
    def input_contract(self) -> MethodInputContract:
        """Return the method-specific input contract."""

    @abstractmethod
    def load_feature_block(self, path: Path) -> FeatureBlock:
        """Load a method input file and return a feature block."""

    def describe(self) -> str:
        """Return a CLI-friendly method description."""
        return self.input_contract().describe()
