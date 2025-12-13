"""
datasets.koch_fractal

Public API for Koch snowflake / Koch annulus 2D distributions used in toy
generative modeling experiments.

Typical usage:
    from datasets.koch_fractal import (
        KochDistributionConfig, KochDistribution,
        generate_koch_2d_dataset, generate_train_test_split,
    )

    dist = KochDistribution(KochDistributionConfig(iterations=5, outer_radius=1.0, inner_radius=0.5))
    X = dist.sample(10000)
"""

from __future__ import annotations

# If your implementation lives in datasets/koch_fractal/koch_fractal.py (recommended),
# this import will work as-is. If the filename differs, update the module name below.
from .koch_fractal import (
    KochSnowflakeConfig,
    KochSnowflake,
    KochDistributionConfig,
    KochDistribution,
    generate_koch_2d_dataset,
    generate_train_test_split,
    create_visualization_svg,
    DataKind,
    Datasets,
)

__all__ = [
    # geometry
    "KochSnowflakeConfig",
    "KochSnowflake",
    # distribution
    "KochDistributionConfig",
    "KochDistribution",
    # convenience generation
    "generate_koch_2d_dataset",
    "generate_train_test_split",
    # visualization
    "create_visualization_svg",
    # enums (if you’re using them elsewhere)
    "DataKind",
    "Datasets",
]
