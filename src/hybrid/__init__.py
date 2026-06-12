"""Hybrid model package - Physics + neural residual TeX inversion.

Requires PyTorch. Importing this package in a non-torch environment raises
ImportError only when hybrid symbols are actually used, not at package load.
"""

try:
    from .residual_model import ResidualMLP, train_residual_model, compute_residual_jacobian
    from .hybrid_model import HybridTeXModel, condition_number_comparison
    __all__ = [
        'ResidualMLP',
        'train_residual_model',
        'compute_residual_jacobian',
        'HybridTeXModel',
        'condition_number_comparison',
    ]
except ImportError:
    # torch not installed — hybrid module unavailable until Mthesis env is activated
    __all__ = []
