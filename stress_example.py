import bamboo as bam
import example_config as ex
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
import numpy as np

'''Run the stress analysis, using cooling simulation data'''
cooling_data = ex.cooled_engine.run_heating_analysis(number_of_points = ex.points, h_gas_model = "3")
stress_data = ex.cooled_engine.run_stress_analysis(cooling_data, ex.wall_material)
max_stress = np.amax(stress_data["thermal_stress"])

'''Get nozzle data'''
shape_x = np.linspace(ex.engine_geometry.x_min, ex.engine_geometry.x_max, ex.points)
shape_y = np.zeros(len(shape_x))

for i in range(len(shape_x)):
    shape_y[i] = ex.engine_geometry.y(shape_x[i])

'''Plot results'''
fig2, ax_s = plt.subplots()

points1 = np.array([shape_x, shape_y]).T.reshape(-1, 1, 2)
points2 = np.array([shape_x, -shape_y]).T.reshape(-1, 1, 2)
segments1 = np.concatenate([points1[:-1], points1[1:]], axis=1)
segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)
# Each element in segments represents a point definining a coloured line segment

if max_stress < ex.wall_material.sigma_y:
    mid = 1.0
    red_max = 0.0
    norm_max = ex.wall_material.sigma_y
else:
    mid = ex.wall_material.sigma_y/max_stress
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
#cbar = fig2.colorbar(line1, ax=ax_s, ticks=[int(ex.wall_material.sigma_y)])
#cbar.set_ticklabels(["$\sigma_y$"])

ax_s.set_xlim(shape_x.min(), shape_x.max())
ax_s.set_ylim(-shape_y.max(), shape_y.max())

max_stress_index = np.where(stress_data["thermal_stress"] == max_stress)
ax_s.axvline(shape_x[max_stress_index], color = 'red', linestyle = '--',
             label = "Max stress {:.1f} MPa, {:.1f}% of $\sigma_y$ ({:.1f} MPa)".format(
             max_stress/10**6, 100*max_stress/ex.wall_material.sigma_y,
             ex.wall_material.sigma_y/10**6))
# Show where the maximum stress in the liner is

plt.gca().set_aspect('equal', adjustable='box')
# Equal axes scales for a true view of the engine

# Required liner thickness from heat flux
P = ex.cooling_jacket.inner_wall.perf_therm
sigma_y = ex.cooling_jacket.inner_wall.sigma_y
stress_array = np.array(stress_data["thermal_stress"])
flux_array = np.array(cooling_data["q_Adot"])
liner_tmax = 2*P*sigma_y/flux_array
# Do not rely on a single pass of heating_analysis to obtain a viable thickness profile
# May implement an iterative approach at some point

q_figs, q_axs = plt.subplots()
qAdot_line = q_axs.plot(cooling_data["x"], cooling_data["q_Adot"], label = "Wall heat transfer", color = 'red')
q_axs.grid()
q_axs.set_xlabel("Position ($m$)")
q_axs.set_ylabel("Heat flux ($W/m^2$)")

t_axs = q_axs.twinx()
tmax_line = t_axs.plot(cooling_data["x"], liner_tmax*1000, label = "Maximum thickness", color = "navy")
t_axs.set_ylabel("Liner thickness ($mm$)")
#print(qAdot_line.get_label())
q_axs.legend(tmax_line + qAdot_line, [l.get_label() for l in (qAdot_line + tmax_line)], loc = "lower left")

ax_s.legend()
plt.show()