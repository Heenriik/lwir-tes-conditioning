import sympy as sp

# Symbolic Planck function (Liou 2002, eq. 1.2.4, p.11) with constants baked in.
# Sympy differentiates and lambdifies once at import; runtime is pure numpy.

# Physical constants
H = 6.626e-34   # Planck constant      [J·s]
C = 3.0e8       # speed of light       [m/s]
K = 1.381e-23   # Boltzmann constant   [J/K]

lam, T = sp.symbols("lam T", positive=True)

# Planck function (sympy)
B = 2 * H * (C)**2 / (lam**5 * (sp.exp(H * C / (lam * K * T)) - 1))

# Lambdified functions for numerical evaluation
planck = sp.lambdify((lam, T), B, "numpy")
dB_dT  = sp.lambdify((lam, T), sp.diff(B, T), "numpy")


def planck_radiance(wavelength, temperature):
    """
    Spectral radiance from Planck's law.

    Ref: Liou (2002) "An Introduction to Atmospheric Radiation", 2nd ed.,
         Academic Press, eq. (1.2.4), p. 11.
         https://ndl.ethernet.edu.et/bitstream/123456789/32124/1/K.%20N.%20Liou.pdf

    Args:
        wavelength (np.ndarray): meters
        temperature (float): Kelvin

    Returns:
        np.ndarray: spectral radiance, W·m⁻³·sr⁻¹
    """
    return planck(wavelength, temperature)


def analytical_planck_derivative(wavelengths, temperature):
    """
    dB/dT — temperature derivative of Planck's law.

    Derived symbolically by sympy at module import (sp.diff of B w.r.t. T) and
    lambdified to numpy. The Planck function itself is from Liou (2002) eq.
    (1.2.4); the derivative is not separately cited — sympy is the source of
    truth, no hand derivation is performed.

    Args:
        wavelengths (np.ndarray): meters
        temperature (float): Kelvin

    Returns:
        np.ndarray: dB/dT, W·m⁻³·sr⁻¹·K⁻¹
    """
    return dB_dT(wavelengths, temperature)


def numerical_planck_derivative(wavelengths, temperature, delta=1e-3):
    """
    Numerical derivative of Planck function wrt temperature. 
    Approximation to validate the analytical version.

    Args:
        wavelengths (np.ndarray)
        temperature (float)
        delta (float): small perturbation

    Returns:
        np.ndarray
    """
    B_plus = planck_radiance(wavelengths, temperature + delta)
    B_minus = planck_radiance(wavelengths, temperature - delta)

    return (B_plus - B_minus) / (2 * delta)