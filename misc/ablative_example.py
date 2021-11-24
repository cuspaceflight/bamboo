import bamboo as bam
import bamboo.cooling as cool
import bamboo.materials

import numpy as np
import matplotlib.pyplot as plt
import pypropep as ppp
import bamboo.plot
import thermo
import time

'''Engine dimensions'''
Ac = np.pi*0.1**2               #Chamber cross-sectional area (m^2)
L_star = 1.5                    #L_star = Volume_c/Area_t
wall_thickness = 2e-3

'''Chamber conditions'''
pc = 15e5               #Chamber pressure (Pa)
mdot = 5.4489           #Mass flow rate (kg/s)
p_amb = 1.01325e5       #Ambient pressure (Pa). 1.01325e5 is sea level atmospheric.
OF_ratio = 3.5          #Oxidiser/fuel mass ratio

'''Get combustion properties from pypropep'''
#Initialise and get propellants
ppp.init()
e = ppp.Equilibrium()
ipa = ppp.PROPELLANTS['ISOPROPYL ALCOHOL']
n2o = ppp.PROPELLANTS['NITROUS OXIDE']

#Add propellants by mass fractions (note the mass fractions can add up to more than 1)
e.add_propellants_by_mass([(ipa, 1),
                           (n2o, OF_ratio)])

#Adiabatic combustion using chamber pressure                      
e.set_state(P = pc/1e5, type='HP')                      

gamma = e.properties.Isex   #I don't know why they use 'Isex' for gamma. 
cp = 1000*e.properties.Cp   #Cp is given in kJ/kg/K, we want J/kg/K
Tc = e.properties.T

'''Choose the models we want to use for transport properties of the exhaust gas'''
thermo_gas = thermo.mixture.Mixture(['N2', 'H2O', 'CO2'], zs = [e.composition['N2'], e.composition['H2O'], e.composition['CO2']])   
gas_transport = cool.TransportProperties(model = "thermo", thermo_object = thermo_gas)

'''Create the engine object'''
perfect_gas = bam.PerfectGas(gamma = gamma, cp = cp)    #Gas for frozen flow
chamber_conditions = bam.ChamberConditions(pc, Tc, mdot)
nozzle = bam.Nozzle.from_engine_components(perfect_gas, chamber_conditions, p_amb, type = "rao", length_fraction = 0.8)
white_dwarf = bam.Engine(perfect_gas, chamber_conditions, nozzle)
chamber_length = L_star*nozzle.At/Ac

'''Add the cooling system to the engine'''
white_dwarf.add_geometry(chamber_length, Ac, wall_thickness)
white_dwarf.add_exhaust_transport(gas_transport)
white_dwarf.add_ablative(bam.materials.Graphite, bam.materials.CopperC700, wall_thickness = 5e-3, regression_rate = 0.0033e-3)

'''Run the heating analysis'''
print(f"Sea level thrust = {white_dwarf.thrust(1e5)/1000} kN")
print(f"Sea level Isp = {white_dwarf.isp(1e5)} s")

cooling_data = white_dwarf.transient_heating_analysis(number_of_points = 10, dt = 0.2, t_max = 20, to_json = False)
white_dwarf.plot_geometry()
bam.plot.animate_transient_temperatures(cooling_data)
plt.show()