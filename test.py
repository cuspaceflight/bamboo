"""
Temporary file for testing bamboo 0.2.0
"""

import bamboo as bam
import numpy as np

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
                                   coolant_transport = bam.materials.CO2, 
                                   type = "vertical")

# Chamber walls

wall1 = bam.Wall(thickness = 1.5e-3, material = bam.materials.Graphite)
wall2 = bam.Wall(thickness = 1.5e-3, material = bam.materials.CopperC106)
wall3 = bam.Wall(thickness = 1.5e-3, material = bam.materials.StainlessSteel304)

engine = bam.Engine(perfect_gas = perfect_gas, 
                    chamber_conditions = chamber_conditions, 
                    geometry = geometry,
                    exhaust_transport = bam.materials.CO2,
                    cooling_jacket = cooling_jacket,
                    walls = [wall1, wall2, wall3],
                    exhaust_convection = "dittus-boelter",
                    coolant_convection = "dittus-boelter")

print(f"Engine set up, throat at x = {engine.geometry.xt} m, mdot = {engine.mdot} kg/s")

results = engine.steady_cooling_simulation(num_grid = 1000)

engine.geometry.plot()
bam.show()

bam.plot.plot_jacket_pressure(results)
bam.show()

bam.plot.plot_temperatures(results)
bam.show()

