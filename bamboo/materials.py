"""Pre-defined materials using the bamboo.cooling.Materials class

References:
    - [1] - Physical and Mechanical Properties of Copper and Copper Alloys, M. Li and S. J. Zinkle, https://core.ac.uk/download/pdf/208176021.pdf   \n
    - [2] - AK Steel 304 / 304L Stainless Steel Product Data Bulletin, https://www.aksteel.com/sites/default/files/2018-01/304304L201706_1.pdf   \n
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

# Ref [1]
CuNiBe_coeffs = [7.71428267e-01,  2.32661906e-03, -8.37447441e-06,  1.38968730e-08, -1.29732074e-11,  4.69949744e-15]
CuNiBe_config = [273.15, 873.15, 6] # Absolute start and end temps, number of coefficients provided above
# Coefficients are given in ascending power order: constant, x, x**2, ...
# Temperatures used with this polynomial must be in Kelvin

# Ref [2]
Steel304_coeffs = [4.11044348e+01, -4.31300681e-01,  1.84371948e-03, -3.89169923e-06,  3.68275665e-09,
                   2.86934185e-13, -3.80214923e-15,  3.10261448e-18, -8.31599675e-22]
Steel304_config = [273.15, 1033.15, 9]

CopperC700 = cool.Material(E = 140e9, sigma_y = 600e6, poisson = 0.355, alpha = 17.5e-6, k = 211, c = 385, rho = 8940, Tsigma_coeffs = CuNiBe_coeffs, Tsigma_range = CuNiBe_config)
Graphite = cool.Material(E = float('NaN'), sigma_y = float('NaN'), poisson = float('NaN'), alpha = float('NaN'), k = 63.81001, c = 0.720)
StainlessSteel304 = cool.Material(E = 193e9, sigma_y = 241e6, poisson = 0.29, alpha = 16e-6, k = 14, c = 500, rho = 8000, Tsigma_coeffs = Steel304_coeffs, Tsigma_range = Steel304_config)