"""Pre-defined materials using the bamboo.cooling.Materials class
"""

import bamboo.cooling as cool

CopperC700 = cool.Material(E = 140e9, sigma_y = 600e6, poisson = 0.355, alpha = 17.5e-6, k = 211, c = 385, rho = 8940)
Graphite = cool.Material(E = float('NaN'), sigma_y = float('NaN'), poisson = float('NaN'), alpha = float('NaN'), k = 63.81001, c = 0.720)
StainlessSteel304 = cool.Material(E = 193e9, sigma_y = 205e6, poisson = float('NaN'), alpha = float('NaN'), k = 16.2, c = 500, rho = float('NaN'))