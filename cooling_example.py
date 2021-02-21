import bamboo as bam
import bamboo.cooling as cool
import numpy as np
import matplotlib.pyplot as plt
import pypropep as ppp
import thermo
import time

'''Chamber conditions'''
Ac = 0.0116666      #Chamber cross-sectional area (m^2)
pc = 10e5           #Chamber pressure (Pa)
mdot = 4.757        #Mass flow rate (kg/s)
p_amb = 0.4e5       #Ambient pressure (Pa). 1.01325e5 is sea level atmospheric.
OF_ratio = 3        #Mass ratio

'''Get combustion properties from pypropep'''
ppp.init()
e = ppp.Equilibrium()
ipa = ppp.PROPELLANTS['ISOPROPYL ALCOHOL']
n2o = ppp.PROPELLANTS['NITROUS OXIDE']
e.add_propellants_by_mass([(ipa, 1), (n2o, OF_ratio)])
e.set_state(P = pc/1e5, type='HP')                      #Adiabatic combustion (enthalpy H is unchanged, P is given)

gamma = e.properties.Isex   #I don't know why they use 'Isex' for gamma. 
cp = 1000*e.properties.Cp   #Cp is given in kJ/kg/K, we want J/kg/K
Tc = e.properties.T

'''Get physical properties of the gas using thermo - use the mole fractions given by pypropep'''
thermo_gas = thermo.mixture.Mixture(['N2', 'H2', 'CO', 'H2O', 'CO2'], 
                                    zs = [e.composition['N2'], e.composition['H2'], e.composition['CO'], e.composition['H2O'], e.composition['CO2']])   

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
thermo_coolant = thermo.chemical.Chemical('Isopropyl Alcohol')

'''Create the engine object'''
perfect_gas = bam.PerfectGas(gamma = gamma, cp = cp)    #Gas for frozen flow
chamber = bam.CombustionChamber(pc, Tc, Ac, mdot)
nozzle = bam.Nozzle.from_engine_components(perfect_gas, chamber, p_amb, type = "rao", length_fraction = 0.8)
white_dwarf = bam.Engine(perfect_gas, chamber, nozzle)

'''Cooling system setup'''
cooling_jacket = cool.CoolingJacket(k_wall, channel_width, channel_height, inlet_T, pc, thermo_coolant, mdot_coolant)
engine_geometry = cool.EngineGeometry(chamber, nozzle, chamber_length, wall_thickness)
cooled_engine = cool.EngineWithCooling(engine_geometry, cooling_jacket, perfect_gas, thermo_gas)

'''Plots'''
#engine_geometry.plot_geometry()
#cooled_engine.show_gas_temperature()

'''Run the cooling system simulation'''
cooling_data = cooled_engine.run_heating_analysis(number_of_points = 1000)

'''Plot the results'''
fig, axs = plt.subplots()
axs.plot(cooling_data["x"], cooling_data["T_wall_inner"] - 273.15, label = "Wall (Inner)")
axs.plot(cooling_data["x"], cooling_data["T_wall_outer"]- 273.15, label = "Wall (Outer)")
axs.plot(cooling_data["x"], cooling_data["T_coolant"]- 273.15, label = "Coolant")
#axs.plot(cooling_data["x"], cooling_data["T_gas"], label = "Exhaust gas")
axs.axvline(cooling_data["boil_off_position"], color = 'red', linestyle = '--', label = "Coolant boil-off")
axs.grid()
axs.set_xlabel("Position (m)")
axs.set_ylabel("Temperature (deg C)")
axs.legend()

plt.show()