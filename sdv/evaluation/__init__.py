"""Module to compare real and synthetic data."""

from sdv.evaluation.evaluation import (
    evaluate_quality,
    run_diagnostic,
)
from sdv.evaluation.utils import (
    get_combination_overlap,
    get_pii_overlap,
    print_referential_integrity,
)


__all__ = (
    'evaluate_quality',
    'run_diagnostic',
    'get_combination_overlap',
    'get_pii_overlap',
    'print_referential_integrity',
)
