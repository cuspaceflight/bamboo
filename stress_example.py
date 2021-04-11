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
Tc = 2800.0         #Chamber temperature (K) - obtained from ProPEP 3
mdot = 4.757        #Mass flow rate (kg/s)
p_amb = 1.01325e5   #Ambient pressure (Pa). 1.01325e5 is sea level atmospheric.
OF_ratio = 3.5      #Oxidiser/fuel mass ratio

'''Engine geometry'''
Ac = np.pi*0.1**2               #Chamber cross-sectional area (m^2)
L_star = 1.5                    #L_star = Volume_c/Area_t
inner_wall_thickness = 2e-3
outer_wall_thickness = 4e-3

'''Coolant jacket'''
inner_wall_material = bam.materials.CopperC700
outer_wall_material = bam.materials.StainlessSteel304
mdot_coolant = mdot/(OF_ratio + 1) 
inlet_T = 298.15  # Coolant inlet temperature
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
engine.add_geometry(chamber_length, Ac, inner_wall_thickness, outer_wall_thickness)
engine.add_exhaust_transport(gas_transport)
engine.add_cooling_jacket(inner_wall_material, outer_wall_material, inlet_T, p_tank, coolant_transport, mdot_coolant,
                          configuration = "vertical", channel_height = 0.001, xs = [-100, 100], blockage_ratio = 0.5)
#engine.add_cooling_jacket(inner_wall_material, outer_wall_material inlet_T, p_tank, coolant_transport, mdot_coolant,
#                          configuration = "spiral", channel_shape = "semi-circle", channel_width = 0.020)

#Add a graphite refractory
engine.add_ablative(bam.materials.Graphite, inner_wall_material, regression_rate = 0.0033e-3, xs = [engine.geometry.x_chamber_end, 100], ablative_thickness = None)

# Run the analyses
cooling_data = engine.steady_heating_analysis(number_of_points=1000)
xs = cooling_data["x"]
steady_stress = engine.run_stress_analysis(heating_result = cooling_data, condition = "steady")
transient_stress = engine.run_stress_analysis(heating_result = cooling_data, condition= "transient", T_amb = inlet_T)

# Graph results
fig1, axs1 = plt.subplots(figsize=(12, 7))
fig2, axs2 = plt.subplots(figsize=(12, 7))

axs1.plot(xs, steady_stress["thermal_stress"]/1E6, label = "Thermal stress")
axs1.plot(xs, steady_stress["yield_adj"]/1E6, label = "Temperature compensated yield stress")
axs1.set_title("Steady state operation: Inner liner")
axs1.set_xlabel("Axial displacement from throat $(m)$")
axs1.set_ylabel("Stress $(MPa)$")
axs1.set_ylim([0, None])
axs1.legend(bbox_to_anchor = (0, -0.16), loc = "lower left")

axs1_2 = axs1.twinx()
axs1_2.plot(xs, steady_stress["deltaT_wall"], color = "red", label = "Coolant side to chamber side $\Delta T$")
axs1_2.set_ylabel("Temperature difference ($\Delta K$)")
axs1_2.set_ylim([0, None])
axs1_2.legend(bbox_to_anchor = (0, -0.22), loc = "lower left")

fig1.subplots_adjust(bottom = 0.16)

axs2.plot(xs, steady_stress["stress_inner_hoop_steady"], label="Inner liner, prior to ignition")
axs2.plot(xs, transient_stress["stress_inner_hoop_II"], label="Inner liner, after ignition")
axs2.plot(xs, steady_stress["stress_outer_hoop"], label="Outer liner")
axs2.set_title("Hoop stresses in liners")

plt.show()