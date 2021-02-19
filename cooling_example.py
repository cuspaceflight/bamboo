import bamboo as bam
import bamboo.cooling as cool
import numpy as np
import thermo
import time

'''Gas properties - obtained from ProPEP 3'''
gamma = 1.264               #Ratio of specific heats cp/cv
molecular_weight = 21.627   #Molecular weight of the exhaust gas (kg/kmol) (only used to calculate R, and hence cp)

'''Chamber conditions'''
Ac = 0.05           #Chamber cross-sectional area (m^2) - made larger because the original number is smaller than my geometry can deal with right now
pc = 10e5           #Chamber pressure (Pa)
Tc = 2458.89        #Chamber temperature (K) - obtained from ProPEP 3
mdot = 4.757        #Mass flow rate (kg/s)
p_amb = 0.4e5       #Ambient pressure (Pa). 1.01325e5 is sea level atmospheric.

'''Create the engine object'''
exhaust_gas = bam.Gas(gamma = gamma, molecular_weight = molecular_weight)
chamber = bam.CombustionChamber(pc, Tc, Ac, mdot)
nozzle = bam.Nozzle.from_engine_components(exhaust_gas, chamber, p_amb, type = "rao", length_fraction = 0.8)
white_dwarf = bam.Engine(exhaust_gas, chamber, nozzle)

'''Chemicals used'''
coolant = thermo.chemical.Chemical('Isopropyl Alcohol')

'''Cooling system geometry'''
cooling_jacket = cool.CoolingJacket(0.01, 0.01)
engine_geometry = cool.EngineGeometry(chamber, nozzle, 0.3)
cooled_engine = cool.EngineWithCooling(engine_geometry, cooling_jacket, coolant, exhaust_gas)

'''Plots'''
#engine_geometry.plot_geometry()
cooled_engine.show_gas_mach()
cooled_engine.show_gas_temperature()