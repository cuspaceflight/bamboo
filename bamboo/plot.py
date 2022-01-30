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

def plot_coolant_pressures(data_dict, plot_static = True, plot_stagnation = True):
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

def plot_coolant_temperatures(data_dict, plot_static = True, plot_stagnation = True):
    """Given the output dictionary from a engine cooling analysis, plot the coolant static and stagnation temperature against postiion

    Args:
        data_dict (dict): Dictionary contaning the cooling analysis results.
        plot_static (bool): Whether or not to plot the static pressure.
        plot_stagnation (bool): Whether or not to plot the stagnation pressure.

    """

    fig, axs = plt.subplots()

    if plot_stagnation == True:
        axs.plot(data_dict["x"], np.array(data_dict["T0_coolant"]), label = "Stagnation temperature")

    if plot_static == True:
        axs.plot(data_dict["x"], np.array(data_dict["T_coolant"]), label = "Static temperature")
        
    axs.legend()
    axs.grid()
    axs.set_xlabel("Axial position (m)")
    axs.set_ylabel("Temperature (K)")

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

def plot_tangential_stress(data_dict, wall_index = 0):
    """Given the output dictionary from a engine cooling analysis, plot the thermal stress in the inner chamber wall, against position. 
    Note you will have to run matplotlib.pyplot.show() or bam.show() to see the plot.

    Args:
        data_dict (dict): Dictionary contaning the cooling analysis results.
        wall_index (int): The index of the wall to plot the stresses for. Defaults to 0 (the exhaust side wall).

    """

    fig, axs = plt.subplots()
    sigma_t_thermal = np.array(data_dict["sigma_t_thermal"])[:, wall_index]
    sigma_t_pressure = np.array(data_dict["sigma_t_pressure"])[:, wall_index]
    sigma_t_max = np.array(data_dict["sigma_t_max"])[:, wall_index]

    axs.plot(data_dict["x"], sigma_t_thermal/1e6, label = "Thermal stress")
    axs.plot(data_dict["x"], sigma_t_pressure/1e6, label = "Pressure stress")
    axs.plot(data_dict["x"], sigma_t_max/1e6, label = "Maximum stress", linestyle = "--")

    axs.legend()
    axs.grid()
    axs.set_xlabel("Axial position (m)")
    axs.set_ylabel("Tangential stress (MPa)")

def plot_coolant_velocity(data_dict):
    """Given the output dictionary from a engine cooling analysis, plot the cooling velocity against axial position. 
    Note you will have to run matplotlib.pyplot.show() or bamboo.show() to see the plot.

    Args:
        data_dict (dict): Dictionary contaning the cooling analysis results.

    """

    fig, axs = plt.subplots()
    axs.plot(data_dict["x"], data_dict["V_coolant"], label = "Coolant velocity")

    axs.legend()
    axs.grid()
    axs.set_xlabel("Axial position (m)")
    axs.set_ylabel(r"Coolant velocity (m s$^{-1}$)")

def plot_coolant_density(data_dict):
    """Plot the cooling density against axial position. 
    Note you will have to run matplotlib.pyplot.show() or bamboo.show() to see the plot.

    Args:
        data_dict (dict): Dictionary contaning the cooling analysis results.

    """

    fig, axs = plt.subplots()
    axs.plot(data_dict["x"], data_dict["rho_coolant"], label = "Coolant density")

    axs.legend()
    axs.grid()
    axs.set_xlabel("Axial position (m)")
    axs.set_ylabel(r"Coolant density (kg m$^{-3}$)")

def plot_thermal_resistances(data_dict, only_indexes = None):
    """Given the output dictionary from an engine cooling analysis, plot the local thermal resistances against position. 
    Note you will have to run matplotlib.pyplot.show() or bamboo.plot.show() to see the plot.

    Args:
        data_dict (dict): Dictionary contaning the cooling analysis results.
        only_indexes (list): List of resistance indexes to plot, e.g. [1, -1] would only show the coolant convective resistance and exhaust convective resistance. Defaults to None, which means everything is plotted.
    """
    fig, ax = plt.subplots()

    R = np.array(data_dict["Rdx"])

    if only_indexes == None:
        for i in range(len(R[0])):
            if i == 0:
                label = "Coolant convection"  
            
            elif i == len(R[0]) - 1 or i == -1:
                label = "Exhaust convection"

            else:
                label = f"Wall {i}"

            ax.plot(data_dict["x"], R[:, i], label = label)

    else:
        for i in only_indexes:
            if i == 0:
                label = "Coolant convection" 
            
            elif i == len(R[0]) - 1 or i == -1:
                label = "Exhaust convection"

            else:
                label = f"Wall {i}"

            ax.plot(data_dict["x"], R[:, i], label = label)  

    ax.grid()
    ax.set_xlabel("Axial position (m)")
    ax.set_ylabel(r"Local thermal resistance (K m W$^{-1}$)")
    ax.legend()

    # Reverse the legend order, so they're arranged in the same order as the lines usually are
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels))

def plot_coolant_h(data_dict):
    """Plot convective heat transfer coefficient for the coolant side.

    Args:
        data_dict (dict): Dictionary contaning the cooling analysis results.
    """
    fig, axs = plt.subplots()

    R = np.array(data_dict["Rdx"]) # Index zero is coolant
    y = np.array(data_dict["y"])
    h = np.zeros(len(R))

    for i in range(len(R)):
        # R = 1 / (hA)
        A = 2 * np.pi * y[i]
        h[i] = 1 / (A * R[i][0])

    axs.plot(data_dict["x"], h, label = "Coolant convection")

    axs.legend()
    axs.grid()
    axs.set_xlabel("Axial position (m)")
    axs.set_ylabel(r"Convective heat transfer coefficient (W m$^{-2}$  K$^{-1}$)")
