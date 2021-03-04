"""Module that provides plotting tools, to streamline the creation of plots.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_temperatures(data_dict, **kwargs):
    """Given the output dictionary from an engine cooling analysis, plot the temperatures against position. 
    Note you will have to run matplotlib.pyplot.show() to see the plot.

    Args:
        data_dict (dict): Dictionary contaning the cooling analysis results.
    
    Keyword Args:
        gas_temperature (bool): If True, the exhaust gas freestream temperatures will be shown.
    """
    fig, ax_T = plt.subplots()
    ax_T.plot(data_dict["x"], np.array(data_dict["T_wall_inner"]) - 273.15, label = "Wall (Inner)")
    ax_T.plot(data_dict["x"], np.array(data_dict["T_wall_outer"])- 273.15, label = "Wall (Outer)")
    ax_T.plot(data_dict["x"], np.array(data_dict["T_coolant"]) - 273.15, label = "Coolant")
    
    if data_dict["boil_off_position"] != None:
        ax_T.axvline(data_dict["boil_off_position"], color = 'red', linestyle = '--', label = "Coolant boil-off")

    if "gas_temperature" in kwargs:
        if kwargs["gas_temperature"] == True:
            ax_T.plot(data_dict["x"], np.array(data_dict["T_gas"]) - 273.15, label = "Exhaust gas")

    ax_T.grid()
    ax_T.set_xlabel("Position (m)")
    ax_T.set_ylabel("Temperature (°C)")
    ax_T.legend()

def plot_h(data_dict, **kwargs):
    """Given the output dictionary from a engine cooling analysis, plot the convective heat transfer coefficients against position. 
    Note you will have to run matplotlib.pyplot.show() to see the plot.

    Args:
        data_dict (dict): Dictionary contaning the cooling analysis results.

    Keyword Args:
        qdot (bool): If True, the heat transfer rate per unit length will also be plotted.
    """
    h_figs, h_axs = plt.subplots()
    h_axs.plot(data_dict["x"], data_dict["h_gas"], label = "Gas")
    h_axs.plot(data_dict["x"], data_dict["h_coolant"], label = "Coolant", )

    if data_dict["boil_off_position"] != None:
        h_axs.axvline(data_dict["boil_off_position"], color = 'red', linestyle = '--', label = "Coolant boil-off")

    if "qdot" in kwargs:
        if kwargs["qdot"] == True:
            q_axs = h_axs.twinx() 
            q_axs.plot(data_dict["x"], data_dict["q_dot"], label = "Heat transfer rate", color = 'red')
            q_axs.legend(loc = "lower left")
            q_axs.set_ylabel("Heat transfer rate (W/m)")

    h_axs.legend()
    h_axs.grid()
    h_axs.set_xlabel("Position (m)")
    h_axs.set_ylabel("Convective Heat Transfer Coefficient (W/m^2/K)")

def plot_qdot(data_dict, **kwargs):
    """Given the output dictionary from a engine cooling analysis, plot the heat transfer rate against position. 
    Note you will have to run matplotlib.pyplot.show() to see the plot.

    Args:
        data_dict (dict): Dictionary contaning the cooling analysis results.

    """

    q_figs, q_axs = plt.subplots()
    q_axs.plot(data_dict["x"], data_dict["q_dot"], label = "Heat transfer rate (W/m)", color = 'red')

    if data_dict["boil_off_position"] != None:
        q_axs.axvline(data_dict["boil_off_position"], color = 'red', linestyle = '--', label = "Coolant boil-off")

    q_axs.legend()
    q_axs.grid()
    q_axs.set_xlabel("Position (m)")
    q_axs.set_ylabel("Heat transfer rate (W/m)")
            