"""Simulation package - Emissivity curve generation, noise addition, calibration, etc."""

from .generator import (
    generate_emissivity_curve, generate_wavelength_sets,
    get_material_emissivity, MATERIAL_LIBRARY, parameter_sweep_grid,
)
from .noise import (
    add_noise, add_gaussian_noise, add_nedt_noise, add_noise_batch, snr_from_nedt,
)
from .calibration import (
    apply_wavelength_shift, apply_wavelength_shift_random,
    apply_fwhm_broadening, calibration_error_study, combined_perturbation_analysis,
)

__all__ = [
    'generate_emissivity_curve', 'generate_wavelength_sets',
    'get_material_emissivity', 'MATERIAL_LIBRARY', 'parameter_sweep_grid',
    'add_noise', 'add_gaussian_noise', 'add_nedt_noise', 'add_noise_batch', 'snr_from_nedt',
    'apply_wavelength_shift', 'apply_wavelength_shift_random',
    'apply_fwhm_broadening', 'calibration_error_study', 'combined_perturbation_analysis',
]