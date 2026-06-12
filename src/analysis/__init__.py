"""Analysis package - Condition number computation, spectral analysis, etc."""

from .conditioning import (
    compute_condition_number,
    sweep_band_count,
    sweep_spectral_placement,
    identifiability_threshold_analysis,
)
from .spectral_analysis import (
    optimal_band_selection,
    emissivity_variability_sweep,
    stability_map_band_snr,
    stability_map_variability_resolution,
)

__all__ = [
    'compute_condition_number',
    'sweep_band_count',
    'sweep_spectral_placement',
    'identifiability_threshold_analysis',
    'optimal_band_selection',
    'emissivity_variability_sweep',
    'stability_map_band_snr',
    'stability_map_variability_resolution',
]