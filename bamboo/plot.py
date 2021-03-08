"""Module that provides plotting tools, to streamline the creation of plots.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

def plot_jacket_pressure(data_dict, **kwargs):
    """Given the output dictionary from a engine cooling analysis, plot the cooling jacket pressure against x position. 
    Note you will have to run matplotlib.pyplot.show() to see the plot.

    Args:
        data_dict (dict): Dictionary contaning the cooling analysis results.

    """

    p_figs, p_axs = plt.subplots()
    p_axs.plot(data_dict["x"], np.array(data_dict["p_coolant"])/1e5, label = "Coolant static pressure (bar)")

    if data_dict["boil_off_position"] != None:
        p_axs.axvline(data_dict["boil_off_position"], color = 'red', linestyle = '--', label = "Coolant boil-off")

    p_axs.legend()
    p_axs.grid()
    p_axs.set_xlabel("Position (m)")
    p_axs.set_ylabel("Coolant pressure (bar)")


def animate_transient_temperatures(data_dict, speed = 1, **kwargs): 
    xs = data_dict["x"]
    ts = data_dict["t"]

    fig = plt.figure()
    ax = plt.axes(xlim=(xs[-1], xs[0]), ylim=(0, 2000))
    ax.grid()
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Temperature (K)")
    ax.set_title(f"t = {ts[0]}")

    T_wall_line, = ax.plot([], [], label = "Wall Temperature")
    ax.legend()

    def init():
        T_wall_line.set_data(xs, data_dict["T_wall"][0])

        return T_wall_line,

    def animate(i):
        T_wall_line.set_data(xs, data_dict["T_wall"][i])
        ax.set_title(f"t = {ts[i]} s")
        return T_wall_line,

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(ts), interval = (ts[1] - ts[0])*1000/speed)
    plt.show()
