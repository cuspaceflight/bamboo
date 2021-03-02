import bamboo as bam
import bamboo.cooling as cool
import numpy as np
import matplotlib.pyplot as plt
import pypropep as ppp
import bamboo.plot
import thermo
import time

'''Engine dimensions'''
Ac = 116.6e-4                   #Chamber cross-sectional area (m^2)
L_star = 1.5                    #L_star = Volume_c/Area_t
wall_thickness = 2e-3

'''Chamber conditions'''
pc = 15e5               #Chamber pressure (Pa)
mdot = 4.757            #Mass flow rate (kg/s)
p_amb = 1.01325e5       #Ambient pressure (Pa). 1.01325e5 is sea level atmospheric.
OF_ratio = 4            #Oxidiser/fuel mass ratio

'''Wall material properties'''
wall_modulus = 140E9 # Young's modulus (Pa)
wall_yield = 600E6 # Yield stress (Pa), 600 MPa is C700 copper alloy, 0.2% plastic
wall_poisson = 0.355 # Poisson's ratio, copper is 0.355
wall_expansion = 17.5E-6 # Thermal expansion coefficient (strain/K), C700 alloy is 17.5E6
wall_conductivity = 211 # Thermal conductivity (W/mK), C700 alloy is 211

'''We want to investigate adding water to the isopropyl alcohol'''
water_mass_fraction = 0.40  #Fraction of the fuel that is water, by mass

'''Coolant jacket'''
mdot_coolant = mdot/(OF_ratio + 1) 
semi_circle_diameter = 4e-3        
inlet_T = 298.15                    #Coolant inlet temperature
thermo_coolant = thermo.mixture.Mixture(['Isopropyl Alcohol', 'Water'], ws = [1 - water_mass_fraction, water_mass_fraction])

'''Get combustion properties from pypropep'''
#Initialise and get propellants
ppp.init()
e = ppp.Equilibrium()
ipa = ppp.PROPELLANTS['ISOPROPYL ALCOHOL']
water = ppp.PROPELLANTS['WATER']
n2o = ppp.PROPELLANTS['NITROUS OXIDE']

#Add propellants by mass fractions (note the mass fractions can add up to more than 1)
e.add_propellants_by_mass([(ipa, 1-water_mass_fraction), 
                           (water, water_mass_fraction), 
                           (n2o, OF_ratio)])

#Adiabatic combustion using chamber pressure                      
e.set_state(P = pc/1e5, type='HP')                      

gamma = e.properties.Isex   #I don't know why they use 'Isex' for gamma. 
cp = 1000*e.properties.Cp   #Cp is given in kJ/kg/K, we want J/kg/K
Tc = e.properties.T

'''Get physical properties of the gas using thermo - use the mole fractions given by pypropep'''
#Exclude H2 and CO- it has very weird Prandtl number effects - it jumps around massively and I'm not sure why.
#Note that if you want carbon monoxide included you must type 'carbon monoxide', not 'CO' - the latter seems to make thermo use methanol (I don't know why)
thermo_gas = thermo.mixture.Mixture(['N2', 'H2O', 'CO2'], zs = [e.composition['N2'], e.composition['H2O'], e.composition['CO2']])   

'''Create the engine object'''
perfect_gas = bam.PerfectGas(gamma = gamma, cp = cp)    #Gas for frozen flow
chamber = bam.ChamberConditions(pc, Tc, mdot)
nozzle = bam.Nozzle.from_engine_components(perfect_gas, chamber, p_amb, type = "rao", length_fraction = 0.8)
white_dwarf = bam.Engine(perfect_gas, chamber, nozzle)
chamber_length = L_star*nozzle.At/Ac

print(f"Sea level thrust = {white_dwarf.thrust(1e5)/1000} kN")
print(f"Sea level Isp = {white_dwarf.isp(1e5)} s")

'''Cooling system setup'''
wall_material = cool.Material(wall_modulus, wall_yield, wall_poisson, wall_expansion, wall_conductivity)
cooling_jacket = cool.CoolingJacket(wall_material, inlet_T, pc, thermo_coolant, mdot_coolant, channel_shape = "semi-circle", circle_diameter = semi_circle_diameter)
engine_geometry = cool.EngineGeometry(nozzle, chamber_length, Ac, wall_thickness)
cooled_engine = cool.EngineWithCooling(chamber, engine_geometry, cooling_jacket, perfect_gas, thermo_gas)

'''Run the cooling system simulation'''
cooling_data = cooled_engine.run_heating_analysis(number_of_points = 1000, h_gas_model = "bartz 2", to_json = "data/heating_output.json")

'''Plot the results'''
#engine_geometry.plot_geometry()
#cooled_engine.show_gas_mach()
bam.plot.plot_temperatures(cooling_data, gas_temperature = False)
bam.plot.plot_qdot(cooling_data)
bam.plot.plot_h(cooling_data, qdot = True)

plt.show()