'''
Subscripts:
    0 - Stagnation condition
    c - Chamber condition (should be the same as stagnation conditions)
    t - At the throat
    e - At the nozzle exit plane
    amb - Atmopsheric/ambient condition
'''
import bamboo as bam
import pypropep as ppp
import numpy as np
import time

'''Chamber conditions'''
Ac = 0.0116666      #Chamber cross-sectional area (m^2)
pc = 10e5           #Chamber pressure (Pa)
mdot = 4.757        #Mass flow rate (kg/s)
p_amb = 0.4e5       #Ambient pressure (Pa). 1.01325e5 is sea level atmospheric.
OF_ratio = 3    #Mass ratio

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

'''Create the engine object using frozen flow'''
perfect_gas = bam.PerfectGas(gamma = gamma, cp = cp)   
chamber = bam.CombustionChamber(pc, Tc, Ac, mdot)
nozzle = bam.Nozzle.from_engine_components(perfect_gas, chamber, p_amb, type = "rao", length_fraction = 0.8)
white_dwarf = bam.Engine(perfect_gas, chamber, nozzle)

print(nozzle)
print(f"Sea level thrust = {white_dwarf.thrust(1e5)/1000} kN")
print(f"Sea level Isp = {white_dwarf.isp(1e5)} s")

nozzle.plot_nozzle()

#Estimate apogee based on apprpoximate Martlet 4 vehicle mass and cross sectional area
apogee_estimate = bam.estimate_apogee(dry_mass = 60, 
                                      propellant_mass = 50, 
                                      engine = white_dwarf, 
                                      cross_sectional_area = 0.03, 
                                      show_plot = False)

print(f"Apogee estimate = {apogee_estimate/1000} km")

#Run an optimisation program to change the nozzle area ratio, to maximise the apogee obtained (I'm not sure if this is working correctly right now).
#white_dwarf.optimise_for_apogee(dry_mass = 60, propellant_mass = 50, cross_sectional_area = 0.03)

#print(white_dwarf.nozzle)
