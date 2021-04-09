import bamboo as bam
import bamboo.cooling as cool
import bamboo.materials

import numpy as np
import pypropep as ppp
import thermo
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap

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
outer_wall_thickness = 5e-3

'''Coolant jacket'''
inner_wall_material = bam.materials.CopperC700
outer_wall_material = bam.materials.StainlessSteel304
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
engine.add_geometry(chamber_length, Ac, inner_wall_thickness, outer_wall_thickness, style="auto")
engine.add_exhaust_transport(gas_transport)
engine.add_cooling_jacket(inner_wall_material, outer_wall_material, inlet_T, p_tank, coolant_transport, mdot_coolant,
                          configuration = "vertical", channel_height = 0.001, xs = [-100, 100], blockage_ratio = 0.5)
#engine.add_cooling_jacket(inner_wall_material, outer_wall_material inlet_T, p_tank, coolant_transport, mdot_coolant,
#                          configuration = "spiral", channel_shape = "semi-circle", channel_width = 0.020)

'''Run a second simulation with an ablative added'''
#Add a graphite refractory.
engine.add_ablative(bam.materials.Graphite, inner_wall_material, regression_rate = 0.0033e-3, xs = [engine.geometry.x_chamber_end, 100], ablative_thickness = None)

print(f"Sea level thrust = {engine.thrust(1e5)/1000} kN")
print(f"Sea level Isp = {engine.isp(1e5)} s")

num_pts = 2000 # Only increased beyond 1000 to make the graph smoother, no meaningful accuracy gain
cooling_data = engine.steady_heating_analysis(number_of_points=num_pts, to_json = "data/heating_output.json")


'''Run a transient stress analysis, using cooling simulation data'''
transient_stress = engine.run_stress_analysis(heating_result=cooling_data, condition="transient")

'''Get nozzle data, including the ablator if it is present, set up transient and steady state figures'''
shape_x = np.linspace(engine.geometry.x_min, engine.geometry.x_max, num_pts)
shape_y = np.zeros(len(shape_x))
fig2, ax_s = plt.subplots()

# Code here involving the ablator is ripped out of Engine.plot_geometry()
x = np.linspace(engine.geometry.x_min, engine.geometry.x_max, num_pts)
ablative_inner = np.zeros(num_pts)
ablative_outer = np.zeros(num_pts)

if engine.has_ablative is True:
    for axes in [ax_s]: # Can add to this if more figures with colour maps are added
        for i in range(len(shape_x)):
            shape_y[i] = engine.geometry.chamber_radius
            ablative_inner[i] = engine.y(x[i], up_to = 'ablative in') - 0.0085 # The -0.0085 corrects for the line width
            ablative_outer[i] = engine.y(x[i], up_to = 'ablative out') - 0.0085 # The -0.0085 corrects for the line width
        axes.fill_between(x, ablative_inner, ablative_outer, color="grey", label = 'Ablative')
        axes.fill_between(x, -ablative_inner, -ablative_outer, color="grey")   
else:
    for i in range(len(shape_x)):
        shape_y[i] = engine.y(shape_x[i])

'''Plot results'''
points1 = np.array([shape_x, shape_y]).T.reshape(-1, 1, 2)
points2 = np.array([shape_x, -shape_y]).T.reshape(-1, 1, 2)
segments1 = np.concatenate([points1[:-1], points1[1:]], axis=1)
segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)
# Each element in segments represents a point defining a coloured line segment

'''Run the steady state stress analysis, using cooling simulation data'''
steady_stress = engine.run_stress_analysis(heating_result=cooling_data, condition="steady")
rel_stress = steady_stress["thermal_stress"]/steady_stress["tadjusted_yield"]
max_rel_stress = np.amax(rel_stress)
min_rel_stress = np.amin(rel_stress)

if max_rel_stress < 1:
    cdict = {"red":  [(0, 0, 0), (0, 0, 0), (1, 0, 0)],
            "green": [(0, 1, 1), (0.7, 0, 0), (1, 0, 0)],
            "blue":  [(0, 0, 0), (0.5, 0, 0), (1, 1, 1)]}
else:
    cdict = {"red":   [(0, 0, 0), (1/(1.1*raw_max), 0, 0), 
                       (1/(raw_max), 1, 1), (1, 1, 1)],
             "green": [(0, 1, 1), (1/(2*raw_max), 0, 0), (1, 0, 0)],
             "blue":  [(0, 0, 0), (1/(2*raw_max), 0, 0),
                       (1/max_rel_stress, 0.4, 0.4), (1, 0, 0)]}  

colours = LinearSegmentedColormap("colours", cdict)
# Set up colour map for stress values

norm = plt.Normalize(0, max_rel_stress)
# Normalise the stress data so it can be mapped

lc1 = LineCollection(segments1, cmap=colours, norm=norm)
lc1.set_array(rel_stress)
lc1.set_linewidth(40)
lc2 = LineCollection(segments2, cmap=colours, norm=norm)
lc2.set_array(rel_stress)
lc2.set_linewidth(40)
# Create line collections defined by segments arrays, map colours

line1 = ax_s.add_collection(lc1)
line2 = ax_s.add_collection(lc2)

cbar = fig2.colorbar(line1, ax=ax_s)
cbar.set_label("Relative stress, $\sigma / \sigma_{ya}$")
#cbar = fig2.colorbar(line1, ax=ax_s, ticks=[int(engine.cooling_jacket.inner_wall.sigma_y)])
#cbar.set_ticklabels(["$\sigma_y$"])

ax_s.set_xlim(shape_x.min(), shape_x.max())
ax_s.set_ylim(-shape_y.max(), shape_y.max())

max_stress_index = np.where(rel_stress == max_rel_stress)[0][0]
# Position of the maximum relative stress given as the index of the corresponding x position from heating analysis

ax_s.axvline(shape_x[max_stress_index], color = 'red', linestyle = '--',
             label = f"""Max relative stress {100*max_rel_stress:.2f}% where """
                     f"""$\sigma = ${steady_stress["thermal_stress"][max_stress_index]/10**6:.2f} $MPa$,"""
                     f""" $\sigma_{{ya}} = $ {steady_stress["tadjusted_yield"][max_stress_index]/10**6:.2f} $MPa$""")

ax_s.set_xlabel("Axial displacement from throat / $m$")
ax_s.set_ylabel("Radial position / $m$")
ax_s.set_title("Steady state thermal stress")
ax_s.set_aspect('equal', adjustable='box')
# Equal axis scales for a true view of the engine
ax_s.legend()
plt.show()