"""Pre-defined materials using the bamboo.cooling.Materials class
"""

import bamboo.cooling as cool

CopperC700 = cool.Material(E = 140E9, sigma_y = 600E6, poisson = 0.355, alpha = 17.5E-6, k = 211)