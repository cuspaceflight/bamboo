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
        show_gas (bool): If True, the exhaust gas freestream temperatures will be shown. Defaults to False.
        show_ablative (bool): If False, the ablative temperatures will not be shown. Defaults to True.
    """
    fig, ax_T = plt.subplots()
    ax_T.plot(data_dict["x"], np.array(data_dict["T_wall_inner"]) - 273.15, label = "Wall (Inner)")
    ax_T.plot(data_dict["x"], np.array(data_dict["T_wall_outer"])- 273.15, label = "Wall (Outer)")
    ax_T.plot(data_dict["x"], np.array(data_dict["T_coolant"]) - 273.15, label = "Coolant")
    
    if data_dict["boil_off_position"] != None:
        ax_T.axvline(data_dict["boil_off_position"], color = 'red', linestyle = '--', label = "Coolant boil-off")

    if "show_gas" in kwargs:
        if kwargs["show_gas"] == True:
            ax_T.plot(data_dict["x"], np.array(data_dict["T_gas"]) - 273.15, label = "Exhaust gas")

    if "show_ablative" in kwargs:
        if kwargs["show_ablative"] == False:
            pass
        else:
            ax_T.plot(data_dict["x"], np.array(data_dict["T_ablative_inner"]) - 273.15, label = "Ablative (inner)")


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

def plot_resistances(data_dict, **kwargs):
    """Given the output dictionary from a engine cooling analysis, plot the thermal resistances of all the components.

    Args:
        data_dict (dict): Dictionary contaning the cooling analysis results.

    """
    figs, axs = plt.subplots()
    axs.plot(data_dict["x"], data_dict["R_gas"], label = "Gas")
    axs.plot(data_dict["x"], data_dict["R_ablative"], label = "Ablative")
    axs.plot(data_dict["x"], data_dict["R_wall"], label = "Wall")
    axs.plot(data_dict["x"], data_dict["R_coolant"], label = "Coolant")

    axs.legend()
    axs.grid()
    axs.set_xlabel("Position (m)")
    axs.set_ylabel("Thermal resistance (K/W)")
 
def plot_exhaust_properties(data_dict, **kwargs):
    """Given the output dictionary from a engine cooling analysis, plot the exhaust gas transport properties

    Args:
        data_dict (dict): Dictionary contaning the cooling analysis results.

    """
    fig, axs = plt.subplots(2,2)
    axs[0,0].plot(data_dict["x"], data_dict["mu_gas"])
    axs[0,0].set_title('Exhaust Gas Absolute Viscosity')
    axs[0,0].set_ylabel('Absolute Viscosity (Pa s)')

    axs[0,1].plot(data_dict["x"], data_dict["k_gas"])
    axs[0,1].set_title('Exhaust Gas Thermal Conductivity')
    axs[0,1].set_ylabel('Thermal Conductivity (W/m/K)')

    axs[1,0].plot(data_dict["x"], data_dict["Pr_gas"])
    axs[1,0].set_title('Exhaust Gas Prandtl Number')
    axs[1,0].set_ylabel('Prandtl Number')

    axs[1,1].set_title('(empty chart)')

    for i in range(len(axs)):
        for j in range(len(axs[i])):
            axs[i, j].grid()
            axs[i, j].set_xlabel("Position (m)")

    fig.tight_layout()

def plot_coolant_properties(data_dict, **kwargs):
    """Given the output dictionary from a engine cooling analysis, plot the coolant transport properties

    Args:
        data_dict (dict): Dictionary contaning the cooling analysis results.

    """
    fig, axs = plt.subplots(2,2)
    axs[0,0].plot(data_dict["x"], data_dict["mu_coolant"])
    axs[0,0].set_title('Coolant Viscosity')
    axs[0,0].set_ylabel('Absolute Viscosity (Pa s)')

    axs[0,1].plot(data_dict["x"], data_dict["k_gas"])
    axs[0,1].set_title('Coolant Thermal Conductivity')
    axs[0,1].set_ylabel('Thermal Conductivity (W/m/K)')

    axs[1,0].plot(data_dict["x"], data_dict["cp_coolant"])
    axs[1,0].set_title('Coolant Specific Heat Capacity')
    axs[1,0].set_ylabel('Specific heat capacity (J/kg/K)')

    axs[1,1].plot(data_dict["x"], data_dict["rho_coolant"])
    axs[1,1].set_title('Coolant Density')
    axs[1,1].set_ylabel('Density (kg/m^3)')

    for i in range(len(axs)):
        for j in range(len(axs[i])):
            axs[i, j].grid()
            axs[i, j].set_xlabel("Position (m)")

    fig.tight_layout()

def animate_transient_temperatures(data_dict, speed = 1, **kwargs): 
    """Animates transient heating analysis data.

    Note:
        Transient analysis modelling is currently incomplete.

    Args:
        data_dict ([type]): [description]
        speed (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """
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
