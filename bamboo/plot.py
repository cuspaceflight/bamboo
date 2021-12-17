"""Module that provides plotting tools, to streamline the creation of plots.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
import numpy as np

def show():
    plt.show()

def plot_temperatures(data_dict, only_indexes = None):
    """Given the output dictionary from an engine cooling analysis, plot the temperatures against position. 
    Note you will have to run matplotlib.pyplot.show() or bamboo.plot.show() to see the plot.

    Args:
        data_dict (dict): Dictionary contaning the cooling analysis results.
        only_indexes (list): List of temperature indexes to plot, e.g. [1, -1] would only show the coolant temperature and exhaust temperature. Defaults to None, which means everything is plotted.
    """
    fig, ax = plt.subplots()

    T = np.array(data_dict["T"])

    if only_indexes == None:
        for i in range(len(T[0])):
            if i == 0:
                label = "Coolant"

            elif i == 1 or i == 1-len(T[0]):
                label = f"Wall (coolant contact)"    
            
            elif i == len(T[0]) - 1 or i == -1:
                label = "Exhaust"

            elif i == len(T[0]) - 2 or i == -2:
                label = "Wall (exhaust contact)"

            else:
                label = f"Wall {i-1}-{i} boundary"

            ax.plot(data_dict["x"], T[:, i], label = label)

    else:
        for i in only_indexes:
            if i == 0:
                label = "Coolant"

            elif i == 1 or i == 1-len(T[0]):
                label = f"Wall (coolant contact)"    
            
            elif i == len(T[0]) - 1 or i == -1:
                label = "Exhaust"

            elif i == len(T[0]) - 2 or i == -2:
                label = "Wall (exhaust contact)"

            else:
                label = f"Wall {i-1}-{i} boundary"

            ax.plot(data_dict["x"], T[:, i], label = label)  

    ax.grid()
    ax.set_xlabel("Axial position (m)")
    ax.set_ylabel("Temperature (K)")
    ax.legend()

    # Reverse the legend order, so they're arranged in the same order as the lines usually are
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels))

def plot_jacket_pressure(data_dict, plot_static = True, plot_stagnation = True, **kwargs):
    """Given the output dictionary from a engine cooling analysis, plot the cooling jacket pressure against x position. 
    Note you will have to run matplotlib.pyplot.show() or bamboo.plot.show() to see the plot.

    Args:
        data_dict (dict): Dictionary contaning the cooling analysis results.
        plot_static (bool): Whether or not to plot the static pressure.
        plot_stagnation (bool): Whether or not to plot the stagnation pressure.

    """

    fig, axs = plt.subplots()

    if plot_stagnation == True:
        axs.plot(data_dict["x"], np.array(data_dict["p0_coolant"])/1e5, label = "Stagnation pressure")

    if plot_static == True:
        axs.plot(data_dict["x"], np.array(data_dict["p_coolant"])/1e5, label = "Static pressure")
        
    axs.legend()
    axs.grid()
    axs.set_xlabel("Axial position (m)")
    axs.set_ylabel("Coolant pressure (bar)")

def plot_q_per_area(data_dict):
    """Given the output dictionary from a engine cooling analysis, plot the heat flux against position. 
    Note you will have to run matplotlib.pyplot.show() or bamboo.show() to see the plot.

    Args:
        data_dict (dict): Dictionary contaning the cooling analysis results.

    """

    fig, axs = plt.subplots()
    axs.plot(data_dict["x"], data_dict["dQ_dA"], label = "Heat flux", color = 'red')

    axs.legend()
    axs.grid()
    axs.set_xlabel("Axial position (m)")
    axs.set_ylabel(r"Radial heat flux (W m$^{-2}$)")


def plot_resistances(data_dict, **kwargs):
    raise ValueError("bamboo.plot.plot_resistances is not yet implemented")
    """Given the output dictionary from a engine cooling analysis, plot the thermal resistances of all the components.

    Args:
        data_dict (dict): Dictionary contaning the cooling analysis results.

    """
    figs, axs = plt.subplots()
    axs.plot(data_dict["x"], data_dict["R_gas"], label = "Gas boundary layer")
    axs.plot(data_dict["x"], data_dict["R_wall"], label = "Wall")
    axs.plot(data_dict["x"], data_dict["R_coolant"], label = "Coolant boundary layer")

    if type(data_dict["R_ablative"][0]) is float("NaN"):
        axs.plot(data_dict["x"], 
                np.array(data_dict["R_gas"]) + np.array(data_dict["R_wall"]) + np.array(data_dict["R_coolant"]), 
                label = "Total resistance", linestyle = '--') 

    else:
        axs.plot(data_dict["x"], data_dict["R_ablative"], label = "Refractory")
        axs.plot(data_dict["x"], 
                np.array(data_dict["R_gas"]) + np.array(data_dict["R_ablative"]) + np.array(data_dict["R_wall"]) + np.array(data_dict["R_coolant"]), 
                label = "Total resistance", linestyle = '--')

    axs.legend()
    axs.grid()
    axs.set_xlabel("Axial position (m)")
    axs.set_ylabel("Thermal resistance (K/W/m)")

def plot_coolant_velocities(data_dict, **kwargs):
    raise ValueError("bamboo.plot.plot_coolant_velocities is not yet implemented")

    """Given the output dictionary from a engine cooling analysis, plot the cooling jacket velocity against x position. 
    Note you will have to run matplotlib.pyplot.show() to see the plot.

    Args:
        data_dict (dict): Dictionary contaning the cooling analysis results.

    """

    fig, axs = plt.subplots()

    axs.plot(data_dict["x"], np.array(data_dict["v_coolant"]), label = "Coolant velocity (m/s)")
        
    axs.legend()
    axs.grid()
    axs.set_xlabel("Axial position (m)")
    axs.set_ylabel("Coolant velocity (m/s)")

def plot_thermal_stress(data_dict, **kwargs):
    raise ValueError("bamboo.plot.plot_thermal_stress is not yet implemented")
    """Given the output dictionary from a engine cooling analysis, plot the thermal stress in the inner chamber wall, against position. 
    Note you will have to run matplotlib.pyplot.show() to see the plot.

    Args:
        data_dict (dict): Dictionary contaning the cooling analysis results.

    """

    fig, axs = plt.subplots()
    axs.plot(data_dict["x"], np.array(data_dict["thermal_stress"])/1e6, label = "Thermal stress")

    axs.grid()
    axs.set_xlabel("Axial position (m)")
    axs.set_ylabel("Thermal stress (MPa)")
