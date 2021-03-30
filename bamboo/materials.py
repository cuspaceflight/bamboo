"""Pre-defined materials using the bamboo.cooling.Materials class

References:
    - [1] - Physical and Mechanical Properties of Copper and Copper Alloys, M. Li and S. J. Zinkle, https://core.ac.uk/download/pdf/208176021.pdf   \n
"""

import bamboo.cooling as cool

# (This may be best placed in the documentation, with a warning at runtime)
# Coefficients for polynomials for temperature ranges, derived from Ref [1] curves use numpy polyfit
# The coefficients are given for unit yield stress - hence they can be applied for other materials but
# this is not recommended, especially for non-copper alloys.
# The curve used is for CuNiBe, temperature hold 2 - it's yield is closest to Copper C700 (CuNi2SiCr)
# Multiple firings of an engine with the same inner liner should be performed with caution.
# Operating at elevated temperatures for extended periods will anneal the material.
# Potentially, the inner liner could yield unexpectedly due to the strength decreasing below the
# maximum transient / thermal stress, so plastic deformation can then occur with further thermal cycles.

CuNiBe_coeffs = [7.71428267e-01,  2.32661906e-03, -8.37447441e-06,  1.38968730e-08, -1.29732074e-11,  4.69949744e-15]
CuNiBe_config = [273.15, 873.15, 6] # Absolute start and end temps, number of coefficients provided above
# Coefficients are given in ascending power order: constant, x, x**2, ...
# Temperatures used with this polynomial must be in Kelvin

CopperC700 = cool.Material(E = 140e9, sigma_y = 600e6, poisson = 0.355, alpha = 17.5e-6, k = 211, c = 385, rho = 8940, Tsigma_coeffs = CuNiBe_coeffs, Tsigma_range = CuNiBe_config)
Graphite = cool.Material(E = float('NaN'), sigma_y = float('NaN'), poisson = float('NaN'), alpha = float('NaN'), k = 63.81001, c = 0.720)
StainlessSteel304 = cool.Material(E = 193e9, sigma_y = 205e6, poisson = float('NaN'), alpha = float('NaN'), k = 16.2, c = 500, rho = float('NaN'))