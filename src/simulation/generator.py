import itertools
import numpy as np

# ---------------------------------------------------------------------------
# HADAR tabulated emissivity library
# Source: Bao et al. 2023, "Heat-assisted detection and ranging", Nature
#         matLib_SyntheticScenes — ground-truth emissivities fed into the
#         Mitsuba renderer; drawn from the NASA JPL ECOSTRESS spectral library
#         (Baldridge et al. 2009, Remote Sens. Environ. 113:711) per Fig. S30
#         of the HADAR supplementary information.
#
# Wavenumber grid: 720:10:1250 cm-1 (54 bands, synthetic scenes).
# Wavelengths converted to metres (ascending): wl_m = 1e-2 / wn_cm-1, reversed.
# Range: 8.00–13.89 µm.
#
# Material mapping (HADAR name → synthetic scene index, thesis name):
#   water       idx 26  water
#   grass       idx 24  vegetation
#   brass       idx 11  metal  (lowest-emissivity measured surface, 0.11–0.15)
#   cinderblock idx  3  concrete  (bark/concrete absent from synthetic scenes;
#                                  cinderblock is the closest mineralogical proxy)
#   soil        idx  8  soil
#   asphalt     idx  1  asphalt
#   human       idx 27  human  (constant 0.95 model value in HADAR)
#   tree        idx 23  tree
# ---------------------------------------------------------------------------

_HADAR_WL_M = np.array([
    8.0000e-06, 8.0645e-06, 8.1301e-06, 8.1967e-06, 8.2645e-06,
    8.3333e-06, 8.4034e-06, 8.4746e-06, 8.5470e-06, 8.6207e-06,
    8.6957e-06, 8.7719e-06, 8.8496e-06, 8.9286e-06, 9.0090e-06,
    9.0909e-06, 9.1743e-06, 9.2593e-06, 9.3458e-06, 9.4340e-06,
    9.5238e-06, 9.6154e-06, 9.7087e-06, 9.8039e-06, 9.9010e-06,
    1.0000e-05, 1.0101e-05, 1.0204e-05, 1.0309e-05, 1.0417e-05,
    1.0526e-05, 1.0638e-05, 1.0753e-05, 1.0870e-05, 1.0989e-05,
    1.1111e-05, 1.1236e-05, 1.1364e-05, 1.1494e-05, 1.1628e-05,
    1.1765e-05, 1.1905e-05, 1.2048e-05, 1.2195e-05, 1.2346e-05,
    1.2500e-05, 1.2658e-05, 1.2821e-05, 1.2987e-05, 1.3158e-05,
    1.3333e-05, 1.3514e-05, 1.3699e-05, 1.3889e-05,
])

_HADAR_EMI = {}

_HADAR_EMI['water'] = np.array([
    0.986652, 0.980933, 0.978241, 0.954945, 0.970262, 0.962999, 0.951553,
    0.954252, 0.951293, 0.952629, 0.943394, 0.952270, 0.950823, 0.938776,
    0.945226, 0.949608, 0.945061, 0.943495, 0.947099, 0.954076, 0.970693,
    0.966088, 0.964135, 0.964972, 0.963913, 0.954742, 0.947306, 0.944785,
    0.954359, 0.951734, 0.952914, 0.954514, 0.950521, 0.957868, 0.953328,
    0.953818, 0.946999, 0.952099, 0.938177, 0.934077, 0.941684, 0.912580,
    0.909412, 0.909430, 0.917345, 0.921504, 0.928652, 0.913496, 0.907306,
    0.904290, 0.890528, 0.876766, 0.863003, 0.849241,
])

_HADAR_EMI['vegetation'] = np.array([
    0.927263, 0.908789, 0.912095, 0.907112, 0.910069, 0.915539, 0.911648,
    0.915771, 0.909639, 0.911899, 0.910889, 0.904858, 0.904385, 0.909733,
    0.905722, 0.902254, 0.901458, 0.902342, 0.902067, 0.901935, 0.900185,
    0.903614, 0.904220, 0.903794, 0.905534, 0.910038, 0.911977, 0.911499,
    0.908467, 0.910870, 0.909676, 0.910251, 0.912760, 0.913102, 0.915262,
    0.916554, 0.917501, 0.915020, 0.917035, 0.915666, 0.914378, 0.914564,
    0.916446, 0.919917, 0.922228, 0.922421, 0.934304, 0.941532, 0.957379,
    0.950000, 0.950000, 0.950000, 0.950000, 0.950000,
])

# Brass — lowest-emissivity surface in HADAR (0.11–0.15).
# Used as "metal" reference to maximise emissivity contrast in conditioning analysis.
_HADAR_EMI['metal'] = np.array([
    0.114462, 0.115676, 0.116855, 0.118912, 0.121176, 0.122985, 0.125176,
    0.128071, 0.130001, 0.133166, 0.134612, 0.134031, 0.133976, 0.135280,
    0.136731, 0.138311, 0.140059, 0.142181, 0.143769, 0.146018, 0.148008,
    0.148800, 0.149019, 0.148692, 0.149378, 0.147226, 0.145472, 0.143590,
    0.141340, 0.138642, 0.135458, 0.133536, 0.131647, 0.131161, 0.128893,
    0.126443, 0.124384, 0.127688, 0.127871, 0.120196, 0.118408, 0.119031,
    0.118008, 0.118003, 0.120112, 0.123568, 0.122962, 0.122101, 0.120247,
    0.118722, 0.116655, 0.117012, 0.117621, 0.118441,
])

# Cinderblock — closest ECOSTRESS proxy to concrete in the synthetic scenes.
# (bark and concrete are absent from matLib_SyntheticScenes; they appear only
# in the experimental scene where emissivities are TES-estimated, not measured.)
_HADAR_EMI['concrete'] = np.array([
    0.915655, 0.915484, 0.915484, 0.915076, 0.918583, 0.918583, 0.920638,
    0.922195, 0.922195, 0.926390, 0.922276, 0.919975, 0.919975, 0.920451,
    0.923732, 0.927282, 0.929036, 0.930693, 0.930693, 0.934895, 0.938452,
    0.939418, 0.940343, 0.942279, 0.943823, 0.946325, 0.949389, 0.952216,
    0.954562, 0.956222, 0.958242, 0.959179, 0.961853, 0.962259, 0.962993,
    0.964505, 0.966236, 0.954055, 0.933690, 0.950566, 0.959332, 0.960566,
    0.962508, 0.965430, 0.965435, 0.961254, 0.963072, 0.962173, 0.965253,
    0.967348, 0.968342, 0.970107, 0.971827, 0.975199,
])

_HADAR_EMI['soil'] = np.array([
    0.950000, 0.963540, 0.941586, 0.832256, 0.860469, 0.828819, 0.776192,
    0.788623, 0.765973, 0.779710, 0.791317, 0.780480, 0.769564, 0.736737,
    0.745955, 0.754441, 0.735660, 0.726721, 0.726577, 0.766120, 0.837744,
    0.837260, 0.840388, 0.849148, 0.845254, 0.831893, 0.834263, 0.843346,
    0.862943, 0.869430, 0.877159, 0.886723, 0.890143, 0.903845, 0.904287,
    0.911092, 0.912877, 0.924026, 0.924240, 0.932162, 0.949584, 0.937088,
    0.943000, 0.952325, 0.969803, 0.947918, 0.969823, 0.948491, 0.954547,
    0.950000, 0.950000, 0.950000, 0.950000, 0.950000,
])

_HADAR_EMI['asphalt'] = np.array([
    0.950171, 0.940856, 0.940856, 0.932446, 0.936136, 0.936136, 0.941114,
    0.946113, 0.946113, 0.954640, 0.949637, 0.947039, 0.947039, 0.946194,
    0.945383, 0.943979, 0.943580, 0.944134, 0.944134, 0.949241, 0.953834,
    0.954913, 0.956034, 0.958141, 0.959060, 0.960586, 0.962004, 0.962695,
    0.963621, 0.964850, 0.966477, 0.966925, 0.966423, 0.966149, 0.965184,
    0.963254, 0.959636, 0.968376, 0.969982, 0.971046, 0.976662, 0.976451,
    0.977381, 0.979882, 0.979677, 0.975173, 0.974315, 0.971343, 0.974728,
    0.976088, 0.976736, 0.978875, 0.983412, 0.975653,
])

# Constant 0.95 — HADAR uses a fixed model value for human skin in all scenes
_HADAR_EMI['human'] = np.full(54, 0.95)

_HADAR_EMI['tree'] = np.array([
    0.960157, 0.959916, 0.961224, 0.959740, 0.960635, 0.959741, 0.960483,
    0.962437, 0.962149, 0.958208, 0.955185, 0.956155, 0.953556, 0.955844,
    0.952293, 0.949695, 0.948905, 0.950552, 0.950167, 0.945544, 0.945841,
    0.944586, 0.941827, 0.940533, 0.935893, 0.938953, 0.936280, 0.938851,
    0.932218, 0.934279, 0.935908, 0.931982, 0.932565, 0.932613, 0.932575,
    0.931724, 0.935247, 0.932527, 0.926146, 0.936466, 0.927213, 0.928423,
    0.943883, 0.943584, 0.937990, 0.948197, 0.945148, 0.953200, 0.944255,
    0.933776, 0.955614, 0.959807, 0.944454, 0.954577,
])

# Pre-computed material library on the HADAR synthetic wavenumber grid
MATERIAL_LIBRARY = {
    name: {
        "wavelengths": _HADAR_WL_M.copy(),
        "emissivity":  emi.copy(),
        "description": name,
    }
    for name, emi in _HADAR_EMI.items()
}


def get_material_emissivity(material_name, wavelengths):
    """
    Interpolate HADAR tabulated emissivity at the requested wavelengths.

    Data source: Bao et al. 2023, Nature — matLib_SyntheticScenes (ECOSTRESS).
    'concrete' uses cinderblock as the closest synthetic-scene proxy
    (bark/concrete absent from synthetic scenes, only TES-estimated in exp. scene).

    Args:
        material_name (str): one of 'water', 'vegetation', 'metal', 'concrete',
            'soil', 'asphalt', 'human', 'tree'
        wavelengths (np.ndarray): metres, shape (N,), within 8.0-13.9 µm

    Returns:
        np.ndarray: emissivity, shape (N,)
    """
    if material_name not in MATERIAL_LIBRARY:
        raise ValueError(
            f"Unknown material '{material_name}'. "
            f"Available: {list(MATERIAL_LIBRARY.keys())}"
        )
    entry = MATERIAL_LIBRARY[material_name]
    return np.interp(wavelengths, entry["wavelengths"], entry["emissivity"])


def generate_emissivity_curve(num_bands, type="smooth", material=None,
                               wavelength_range=None, variability=0.0, seed=None):
    """
    Generate an emissivity spectrum.

    When `material` is given, uses the HADAR tabulated library and ignores `type`.
    `variability` adds N(0, variability) noise and clips to [0.01, 1.0].

    Args:
        num_bands (int): number of spectral bands
        type (str): 'smooth' or 'random' (used only when material is None)
        material (str or None): 'water', 'vegetation', 'metal', 'concrete',
            'soil', 'asphalt', 'human', 'tree'
        wavelength_range (tuple or None): (wl_min, wl_max) in metres;
            defaults to (8e-6, 14e-6)
        variability (float): std of additive Gaussian noise
        seed (int or None): random seed

    Returns:
        np.ndarray: emissivity, shape (num_bands,)
    """
    rng = np.random.default_rng(seed)
    wl_min, wl_max = wavelength_range if wavelength_range else (8e-6, 14e-6)
    wavelengths = np.linspace(wl_min, wl_max, num_bands)

    if material is not None:
        emissivity = get_material_emissivity(material, wavelengths)
    elif type == "smooth":
        emissivity = np.clip(np.sin(np.linspace(0, 3, num_bands)) * 0.1 + 0.9, 0, 1)
    elif type == "random":
        emissivity = rng.uniform(0.7, 1.0, num_bands)
    else:
        raise ValueError(f"Unknown emissivity type '{type}'")

    if variability > 0:
        emissivity = emissivity + rng.normal(0, variability, num_bands)
        emissivity = np.clip(emissivity, 0.01, 1.0)

    return emissivity


def generate_wavelength_sets(min_wl=8e-6, max_wl=14e-6,
                              band_counts=None, n_bands=None,
                              mode="uniform", strategy=None,
                              target_wavelengths=None, seed=None):
    """
    Generate spectral sampling configurations.

    Accepts either `band_counts` (list) or `n_bands` (single int).
    `strategy` is an alias for `mode`; 'targeted' uses `target_wavelengths`.

    Args:
        min_wl, max_wl (float): wavelength range in metres
        band_counts (list[int] or None): list of band counts -> returns list of arrays
        n_bands (int or None): single band count -> returns single array
        mode (str): 'uniform', 'random', 'clustered', 'targeted'
        strategy (str or None): alias for mode; takes precedence if given
        target_wavelengths (list[float] or None): preferred wavelengths for 'targeted'
        seed (int or None): random seed

    Returns:
        list[np.ndarray] when band_counts is given; np.ndarray when n_bands is given
    """
    rng = np.random.default_rng(seed)
    effective_mode = strategy if strategy is not None else mode

    if n_bands is not None:
        return _single_wavelength_set(n_bands, min_wl, max_wl, effective_mode,
                                      target_wavelengths, rng)

    if band_counts is None:
        band_counts = [3, 5, 10, 20, 50]

    return [
        _single_wavelength_set(n, min_wl, max_wl, effective_mode, target_wavelengths, rng)
        for n in band_counts
    ]


def _single_wavelength_set(n, min_wl, max_wl, mode, target_wavelengths, rng):
    if mode == "uniform":
        return np.linspace(min_wl, max_wl, n)

    elif mode == "random":
        return np.sort(rng.uniform(min_wl, max_wl, n))

    elif mode == "targeted":
        if target_wavelengths is None:
            # Aligned the defaults with ASTER TIR centers explicitly: [8.30, 8.65, 9.10, 10.60, 11.30] µm from 
            # Gillespie et al. (1998), Table I.
            # target_wavelengths = [9.6e-6, 10.5e-6, 11.5e-6]
            target_wavelengths = [8.30e-6, 8.65e-6, 9.10e-6, 10.60e-6, 11.30e-6]
        targets = np.array(target_wavelengths)
        targets = targets[(targets >= min_wl) & (targets <= max_wl)]
        n_fill = max(0, n - len(targets))
        fill = np.linspace(min_wl, max_wl, n_fill + 2)[1:-1] if n_fill > 0 else np.array([])
        wl = np.sort(np.concatenate([targets, fill]))
        return wl[:n]

    else:
        raise ValueError(f"Unknown wavelength mode '{mode}'")


def parameter_sweep_grid(T_values, materials, band_counts, strategies):
    """
    Cartesian product of experiment parameters.

    Returns:
        list[dict]: each dict is one experimental configuration with keys
            T, material, n_bands, strategy
    """
    configs = []
    for T, mat, n, strat in itertools.product(T_values, materials, band_counts, strategies):
        configs.append({"T": T, "material": mat, "n_bands": n, "strategy": strat})
    return configs
