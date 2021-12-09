"""
Temporary file for testing bamboo 0.2.0
"""

import bamboo as bam
import numpy as np

mdot_coolant = 0.2          # Coolant mass flow rate (kg/s)

# Combustion chamber
perfect_gas = bam.PerfectGas(gamma = 1.31, cp = 830)
chamber_conditions = bam.ChamberConditions(p0 = 10e5, T0 = 2800)

# Engine contour
xs = [-0.150, -0.050, 0.0, 0.025]
ys = [0.045, 0.045, 0.02, 0.025]

geometry = bam.Geometry(xs = xs, ys = ys)

# Cooling jacket
cooling_jacket = bam.CoolingJacket(T_coolant_in = 298, 
                                   p0_coolant_in = 2e5, 
                                   mdot_coolant = 0.2, 
                                   channel_height = 2e-3, 
                                   coolant_transport = bam.materials.Water, 
                                   type = "vertical")

# Chamber walls
wall = bam.Wall(thickness = 1.5e-3, material = bam.materials.CopperC106)

engine = bam.Engine(perfect_gas = perfect_gas, 
                    chamber_conditions = chamber_conditions, 
                    geometry = geometry,
                    exhaust_transport = bam.materials.CO2,
                    cooling_jacket = cooling_jacket,
                    walls = wall,
                    exhaust_convection = "dittus-boelter",
                    coolant_convection = "dittus-boelter")

print(f"Engine set up, throat at x = {engine.geometry.xt} m")

engine.steady_cooling_simulation()