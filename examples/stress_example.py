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
Ac = np.pi*0.1**2   #Chamber cross-sectional area (m^2)
L_star = 1.5        #L_star = Volume_c/Area_t
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
engine.add_cooling_jacket(wall_material, inlet_T, p_tank, coolant_transport, mdot_coolant, configuration = "spiral", channel_shape = "semi-circle", channel_width = 0.020)

print(f"Sea level thrust = {engine.thrust(1e5)/1000} kN")
print(f"Sea level Isp = {engine.isp(1e5)} s")     


cooling_data = engine.steady_heating_analysis(to_json = "data/heating_output.json")

'''Run the stress analysis, using cooling simulation data'''
stress_data = engine.run_stress_analysis(cooling_data, wall_material)
max_stress = np.amax(stress_data["thermal_stress"])

'''Get nozzle data'''
shape_x = np.linspace(engine.geometry.x_min, engine.geometry.x_max, 1000)
shape_y = np.zeros(len(shape_x))

for i in range(len(shape_x)):
    shape_y[i] = engine.y(shape_x[i])

'''Plot results'''
fig2, ax_s = plt.subplots()

points1 = np.array([shape_x, shape_y]).T.reshape(-1, 1, 2)
points2 = np.array([shape_x, -shape_y]).T.reshape(-1, 1, 2)
segments1 = np.concatenate([points1[:-1], points1[1:]], axis=1)
segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)
# Each element in segments represents a point definining a coloured line segment


if max_stress < engine.cooling_jacket.inner_wall.sigma_y:
    mid = 1.0
    red_max = 0.0
    norm_max = engine.cooling_jacket.inner_wall.sigma_y
else:
    mid = engine.cooling_jacket.inner_wall.sigma_y/max_stress
    norm_max = max_stress
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

norm = plt.Normalize(np.amin(stress_data["thermal_stress"]), norm_max)
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
#cbar = fig2.colorbar(line1, ax=ax_s, ticks=[int(engine.cooling_jacket.inner_wall.sigma_y)])
#cbar.set_ticklabels(["$\sigma_y$"])

ax_s.set_xlim(shape_x.min(), shape_x.max())
ax_s.set_ylim(-shape_y.max(), shape_y.max())

max_stress_index = np.where(stress_data["thermal_stress"] == max_stress)
ax_s.axvline(shape_x[max_stress_index], color = 'red', linestyle = '--',
             label = "Max stress {:.1f} MPa, {:.1f}% of $\sigma_y$ ({:.1f} MPa)".format(
             max_stress/10**6, 100*max_stress/engine.cooling_jacket.inner_wall.sigma_y,
             engine.cooling_jacket.inner_wall.sigma_y/10**6))
# Show location of maximum inner line stress

plt.gca().set_aspect('equal', adjustable='box')
# Equal axes scales for a true view of the engine

ax_s.legend()
plt.show()