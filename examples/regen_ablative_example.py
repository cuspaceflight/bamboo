'''
Subscripts:
    0 - Stagnation condition
    c - Chamber condition (should be the same as stagnation conditions)
    t - At the throat
    e - At the nozzle exit plane
    amb - Atmopsheric/ambient condition
'''
import bamboo as bam
import bamboo.cooling as cool
import numpy as np
import time
import thermo
import matplotlib.pyplot as plt

'''Gas properties - obtained from ProPEP 3'''
gamma = 1.264               #Ratio of specific heats cp/cv
molecular_weight = 21.627   #Molecular weight of the exhaust gas (kg/kmol) (only used to calculate R, and hence cp)

'''Engine operating points'''
p_tank = 25e5       #Tank / inlet coolant stagnation pressure (Pa) - used for cooling jacket
pc = 10e5           #Chamber pressure (Pa)
Tc = 2800.0           #Chamber temperature (K) - obtained from ProPEP 3
mdot = 4.757        #Mass flow rate (kg/s)
p_amb = 1.01325e5   #Ambient pressure (Pa). 1.01325e5 is sea level atmospheric.
OF_ratio = 3.5      #Oxidiser/fuel mass ratio

'''Engine geometry'''
Ac = np.pi*0.1**2               #Chamber cross-sectional area (m^2)
L_star = 1.5                    #L_star = Volume_c/Area_t
wall_thickness = 2e-3

'''Coolant jacket'''
wall_material = bam.materials.CopperC700
mdot_coolant = mdot/(OF_ratio + 1) 
inlet_T = 298.15                    #Coolant inlet temperature
thermo_coolant = thermo.chemical.Chemical('isopropanol')
coolant_transport = cool.TransportProperties(model = "thermo", thermo_object = thermo_coolant, force_phase = 'l')

'''Create the engine object'''
perfect_gas = bam.PerfectGas(gamma = gamma, molecular_weight = molecular_weight)
chamber = bam.ChamberConditions(pc, Tc, mdot)
nozzle = bam.Nozzle.from_engine_components(perfect_gas, chamber, p_amb, type = "rao", length_fraction = 0.8)
engine = bam.Engine(perfect_gas, chamber, nozzle)
chamber_length = L_star*nozzle.At/Ac

'''Choose the models we want to use for transport properties of the exhaust gas'''
thermo_gas = thermo.mixture.Mixture(['N2', 'H2O', 'CO2'], ws = [0.49471, 0.14916, 0.07844])            #Very crude exhaust gas model - using weight fractions from CEA.
gas_transport = cool.TransportProperties(model = "thermo", thermo_object = thermo_gas, force_phase = 'g')

'''Cooling system setup'''
engine.add_geometry(chamber_length, Ac, wall_thickness)
engine.add_exhaust_transport(gas_transport)
#engine.add_cooling_jacket(wall_material, inlet_T, p_tank, coolant_transport, mdot_coolant, configuration = "vertical", channel_height = 0.001, xs = [-100, 100])
engine.add_cooling_jacket(wall_material, inlet_T, p_tank, coolant_transport, mdot_coolant, configuration = "spiral", channel_shape = "semi-circle", channel_width = 0.020)

'''Run a steady state simulation'''
#data = engine.steady_heating_analysis()
#bam.plot.plot_temperatures(data)

'''Run a second simulation with an ablative added'''
#Add a graphite refractory, and override the copper cooling jacket with a stainless steel layer.
engine.add_ablative(bam.materials.Graphite, bam.materials.StainlessSteel304, regression_rate = 0.0033e-3, xs = [engine.geometry.x_chamber_end, 100], ablative_thickness = None)

print(nozzle)

data = engine.steady_heating_analysis()
bam.plot.plot_temperatures(data)
bam.plot.plot_jacket_pressure(data)

bam.plot.plot_resistances(data)
bam.plot.plot_exhaust_properties(data)
bam.plot.plot_coolant_properties(data)

engine.plot_geometry()
plt.show()
