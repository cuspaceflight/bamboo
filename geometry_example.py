'''
Subscripts:
    0 - Stagnation condition
    c - Chamber condition (should be the same as stagnation conditions)
    t - At the throat
    e - At the nozzle exit plane
    amb - Atmopsheric/ambient condition
'''
import bamboo as bam
import numpy as np
import time
import matplotlib.pyplot as plt

'''Gas properties - obtained from ProPEP 3'''
gamma = 1.264               #Ratio of specific heats cp/cv
molecular_weight = 21.627   #Molecular weight of the exhaust gas (kg/kmol) (only used to calculate R, and hence cp)

'''Chamber conditions'''
pc = 10e5           #Chamber pressure (Pa)
Tc = 2458.89        #Chamber temperature (K) - obtained from ProPEP 3
mdot = 4.757        #Mass flow rate (kg/s)
p_amb = 1.01325e5   #Ambient pressure (Pa). 1.01325e5 is sea level atmospheric.

'''Create the engine object'''
perfect_gas = bam.PerfectGas(gamma = gamma, molecular_weight = molecular_weight)
chamber = bam.ChamberConditions(pc, Tc, mdot)
nozzle = bam.Nozzle.from_engine_components(perfect_gas, chamber, p_amb, type = "rao", length_fraction = 0.8)
white_dwarf = bam.Engine(perfect_gas, chamber, nozzle)

'''Add extra geometry'''
Ac = np.pi*0.1**2                         #Chamber cross-sectional area (m^2)
L_star = 1.5                              #L_star = Volume_c/Area_t
wall_thickness = [1e-3, 50e-3]      #Thicker walls as exit than entrance
chamber_length = L_star*nozzle.At/Ac

white_dwarf.add_geometry(chamber_length, Ac, wall_thickness)

white_dwarf.plot_geometry()
plt.show()