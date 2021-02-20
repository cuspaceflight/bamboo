import bamboo as bam
import bamboo.cooling as cool
import numpy as np
import matplotlib.pyplot as plt
import thermo
import time

'''Gas properties - obtained from ProPEP 3'''
gamma = 1.264               #Ratio of specific heats cp/cv
molecular_weight = 21.627   #Molecular weight of the exhaust gas (kg/kmol) (only used to calculate R, and hence cp)

'''Chamber conditions'''
Ac = 0.05           #Chamber cross-sectional area (m^2) - made larger because the original number is smaller than my geometry can deal with right now
pc = 10e5           #Chamber pressure (Pa)
Tc = 2458.89        #Chamber temperature (K) - obtained from ProPEP 3 with an OF ratio = 3
mdot = 4.757        #Mass flow rate (kg/s)
p_amb = 0.4e5       #Ambient pressure (Pa). 1.01325e5 is sea level atmospheric.

thermo_gas = thermo.mixture.Mixture(['N2', 'H2', 'CO','H2O', 'CO2'], zs = [0.368, 0.260, 0.178, 0.0984, 1-0.368-0.260-0.178-0.0984])    #Apprximately from results given by ProPEP 3

'''Engine dimensions'''
chamber_length = 75e-2
wall_thickness = 2e-3

'''Coolant jacket'''
OF_mass_ratio = 3
mdot_coolant = mdot/(OF_mass_ratio + 1)
channel_width = 1e-3
channel_height = 1e-3
k_wall = 300
inlet_T = 298.15    #Coolant inlet temperature
coolant = thermo.chemical.Chemical('Isopropyl Alcohol')

'''Create the engine object'''
exhaust_gas = bam.Gas(gamma = gamma, molecular_weight = molecular_weight)
chamber = bam.CombustionChamber(pc, Tc, Ac, mdot)
nozzle = bam.Nozzle.from_engine_components(exhaust_gas, chamber, p_amb, type = "rao", length_fraction = 0.8)
white_dwarf = bam.Engine(exhaust_gas, chamber, nozzle)

'''Cooling system setup'''
cooling_jacket = cool.CoolingJacket(k_wall, channel_width, channel_height, inlet_T, pc, coolant, mdot_coolant)
engine_geometry = cool.EngineGeometry(chamber, nozzle, chamber_length, wall_thickness)
cooled_engine = cool.EngineWithCooling(engine_geometry, cooling_jacket, exhaust_gas, thermo_gas)

'''Plots'''
#engine_geometry.plot_geometry()
#cooled_engine.show_gas_temperature()

'''Run the cooling system simulation'''
cooling_data = cooled_engine.run_heating_analysis(number_of_points = 1000)

'''Plot the results'''
plt.plot(cooling_data["x"], cooling_data["T_wall_inner"] - 273.15, label = "Wall (Inner)")
plt.plot(cooling_data["x"], cooling_data["T_wall_outer"]- 273.15, label = "Wall (Outer)")
plt.plot(cooling_data["x"], cooling_data["T_coolant"]- 273.15, label = "Coolant")
#plt.plot(cooling_data["x"], cooling_data["T_gas"], label = "Exhaust gas")
plt.grid()
plt.xlabel("Position (m)")
plt.ylabel("Temperature (deg C)")
plt.legend()
plt.show()