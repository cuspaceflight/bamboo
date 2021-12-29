"""Simple engine example, but in .py format instead of Jupyter notebook
"""

# Import required modules
import bamboo as bam

# Chamber conditions
pc = 10e5                   # Chamber pressure (Pa)
p_amb = 1e5                 # Ambient pressure (Pa) - we'll use sea level atmospheric
Tc = 2800                   # Combustion chamber temperature (K) - would normally get this from an equilibrium program (e.g. Cantera, NASA CEA, ProPEP).

# Geometry properties
Rc = 0.045                  # Chamber radius (m)
Rt = 0.02                   # Throat radius (m)
area_ratio = 4              # Area ratio (A_exit / A_throat)
Lc = 0.10                   # Length of chamber (up to beginning of nozzle converging section) (m)
theta_conv = 45             # Angle of converging section (deg)

# Use the in-built Rao geometry generator
xs, ys = bam.rao.get_rao_contour(Rc = Rc, 
                                 Rt = Rt, 
                                 area_ratio = area_ratio, 
                                 Lc = Lc, 
                                 theta_conv = theta_conv)

# Set up the objects we need
perfect_gas = bam.PerfectGas(gamma = 1.31, cp = 830)                   # Approximate values for CO2
chamber_conditions = bam.ChamberConditions(p0 = pc, T0 = Tc)
geometry = bam.Geometry(xs = xs, ys = ys)
exhaust_transport = bam.materials.CO2                                  # Use the built-in CO2 approximation
wall = bam.Wall(material = bam.materials.CopperC106, thickness = 2e-3) # Use the built in C106 copper data

# Submit them all at inputs to the Engine object
engine = bam.Engine(perfect_gas = perfect_gas, 
                    chamber_conditions = chamber_conditions, 
                    geometry = geometry,
                    exhaust_transport = exhaust_transport,
                    walls = wall)

# Cooling jacket properties
inlet_T = 298.15                           # Coolant inlet temperature (K)
inlet_p0 = 30e5                            # Coolant inlet stagnation pressure (bar)
OF_ratio = 3.5                             # Oxidiser/fuel mass ratio
mdot_coolant = 0.2                         # Coolant mass flow rate (kg/s)

# Add a spiral cooling jacket to the engine
cooling_jacket = bam.CoolingJacket(T_coolant_in = inlet_T, 
                                  p0_coolant_in = inlet_p0, 
                                  mdot_coolant = mdot_coolant, 
                                  channel_height = 2e-3,
                                  blockage_ratio = 0.3,
                                  number_of_fins = 100,
                                  coolant_transport = bam.materials.Water,   # Use bamboo's built-in water approximation
                                  configuration = 'vertical')

# Attach the cooling jacket
engine.cooling_jacket = cooling_jacket

# Plot the engine to scale
engine.plot()
bam.show()