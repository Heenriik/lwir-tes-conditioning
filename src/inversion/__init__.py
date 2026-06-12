"""Inversion package - Jacobian computation, parameter inversion, etc."""

# Explicit imports for submodules
from .jacobian import (
    compute_jacobian, compute_jacobian_numerical,
    jacobian_svd, validate_jacobian, parameter_sensitivity, scale_jacobian
)
from .tex_inversion import invert_tex, inversion_ensemble

# Define what gets imported with 'from inversion import *'
__all__ = [
    'compute_jacobian', 'compute_jacobian_numerical',
    'jacobian_svd', 'validate_jacobian', 'parameter_sensitivity', 'scale_jacobian',
    'invert_tex', 'inversion_ensemble',
]
