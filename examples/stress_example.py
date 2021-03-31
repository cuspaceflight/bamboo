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
mdot = 5.757        #Mass flow rate (kg/s)
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

'''Run a second simulation with an ablative added'''
#Add a graphite refractory, and override the copper cooling jacket with a stainless steel layer.
engine.add_ablative(bam.materials.Graphite, bam.materials.StainlessSteel304, regression_rate = 0.0033e-3, xs = [engine.geometry.x_chamber_end, 100], ablative_thickness = None)

print(f"Sea level thrust = {engine.thrust(1e5)/1000} kN")
print(f"Sea level Isp = {engine.isp(1e5)} s")     

num_pts = 4000 # Only increased beyond 1000 to make the graph smoother, no meaningful accuracy gain
cooling_data = engine.steady_heating_analysis(number_of_points=num_pts, to_json = "data/heating_output.json")

'''Run the stress analysis, using cooling simulation data'''
stress_data = engine.run_stress_analysis(cooling_data, wall_material)
max_rel_stress = np.amax(stress_data["thermal_stress"]/stress_data["tadjusted_yield"])
min_rel_stress = np.amin(stress_data["thermal_stress"]/stress_data["tadjusted_yield"])

'''Get nozzle data'''
shape_x = np.linspace(engine.geometry.x_min, engine.geometry.x_max, num_pts)
shape_y = np.zeros(len(shape_x))

for i in range(len(shape_x)):
    shape_y[i] = engine.y(shape_x[i])

'''Plot results'''
fig2, ax_s = plt.subplots()

points1 = np.array([shape_x, shape_y]).T.reshape(-1, 1, 2)
points2 = np.array([shape_x, -shape_y]).T.reshape(-1, 1, 2)
segments1 = np.concatenate([points1[:-1], points1[1:]], axis=1)
segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)
# Each element in segments represents a point defining a coloured line segment

norm_max = max_rel_stress
if max_rel_stress < np.amax(stress_data["tadjusted_yield"]):
    mid = 1.0
    red_max = 0.0
else:
    mid = engine.cooling_jacket.inner_wall.sigma_y/max_stress
    red_max = 1.0
# Check if yield is reached to adjust colour mapping, normalisation
   
cdict = {"red":  [(0.0, 0.0, 0.0),
                  (mid, 0.0, 0.0),
                  (1.0, red_max, red_max)],  
    
        "green": [(0.0, 1.0, 1.0),
                  (mid/2, 0.0, 0.0),
                  (1.0, 0.0, 0.0)],
         
        "blue":  [(0.0, 0.0, 0.0),
                  (mid/2, 0.0, 0.0),
                  (mid, 1.0, 1.0),
                  (1.0, 0.0, 0.0)]}
colours = LinearSegmentedColormap("colours", cdict)
# Set up colour map for stress values

norm = plt.Normalize(np.amin(stress_data["thermal_stress"]/min_rel_stress), norm_max)
# Normalise the stress data so it can be mapped

lc1 = LineCollection(segments1, cmap=colours, norm=norm)
lc1.set_array(stress_data["thermal_stress"])
lc1.set_linewidth(20)
lc2 = LineCollection(segments2, cmap=colours, norm=norm)
lc2.set_array(stress_data["thermal_stress"])
lc2.set_linewidth(20)
# Create line collections defined by segments arrays, map colours

line1 = ax_s.add_collection(lc1)
line2 = ax_s.add_collection(lc2)

cbar = fig2.colorbar(line1, ax=ax_s)
cbar.set_label("$\sigma$ / $Pa$")
#cbar = fig2.colorbar(line1, ax=ax_s, ticks=[int(engine.cooling_jacket.inner_wall.sigma_y)])
#cbar.set_ticklabels(["$\sigma_y$"])

ax_s.set_xlim(shape_x.min(), shape_x.max())
ax_s.set_ylim(-shape_y.max(), shape_y.max())

max_stress_index = np.where(stress_data["thermal_stress"]/stress_data["tadjusted_yield"] == max_rel_stress)[0][0]
# Position of the maximum relative stress given as the index of the corresponding x position from heating analysis
yield_max_rel = stress_data["tadjusted_yield"][max_stress_index]
# The local temperature adjusted yield stress at the location of the highest relative stress

ax_s.axvline(shape_x[max_stress_index], color = 'red', linestyle = '--',
             label = f"""Max relative stress {100*max_rel_stress:.2f}% where """
                     f"""$\sigma = $ {stress_data["thermal_stress"][max_stress_index]/10**6:.2f} $MPa$,"""
                     f""" $\sigma_y = $ {yield_max_rel/10**6:.2f} $MPa$""")

plt.figtext(0.5, 0.12, "(Ablator / refractory not currently shown, if present)", ha="center", fontsize=10)

ax_s.set_xlabel("Axial displacement from throat / $m$")
ax_s.set_ylabel("Radial position / $m$")
plt.gca().set_aspect('equal', adjustable='box')
# Equal axes scales for a true view of the engine

ax_s.legend()
plt.show()