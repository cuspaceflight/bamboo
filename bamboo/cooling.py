'''
Extra tools for modelling the cooling system of a liquid rocket engine.

Room for improvement:
    - My equation for h_gas is the less accurate version, with the Bartz correction factors (this was just to avoid needing the extra parameters for the Bartz equation)
    - The EngineWithCooling.rho() function calculates rho by doing p/RT, but it would probably be faster to just use isentropic compressible flow relations.

Useful:
    - List of CoolProp properties: http://www.coolprop.org/coolprop/HighLevelAPI.html#table-of-string-inputs-to-propssi-function
References:
    - [1] - The Thrust Optimised Parabolic nozzle, AspireSpace, http://www.aspirespace.org.uk/downloads/Thrust%20optimised%20parabolic%20nozzle.pdf   \n
    - [2] - Rocket Propulsion Elements, 7th Edition  \n
    - [3] - Regenerative cooling of liquid rocket engine thrust chambers, ASI, https://www.researchgate.net/profile/Marco-Pizzarelli/publication/321314974_Regenerative_cooling_of_liquid_rocket_engine_thrust_chambers/links/5e5ecd824585152ce804e244/Regenerative-cooling-of-liquid-rocket-engine-thrust-chambers.pdf  \n
'''

import bamboo as bam
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
import scipy
import json


SIGMA = 5.670374419e-8      #Stefan-Boltzmann constant (W/m^2/K^4)


def black_body(T):
    """Get the black body radiation emitted over a hemisphere, at a given temperature.

    Args:
        T (float): Temperature of the body (K)

    Returns:
        float: Radiative heat transfer rate, per unit emitting area on the body (W/m^2)
    """
    return SIGMA*T**4

def h_gas_1(D, M, T, rho, gamma, R, mu, k, Pr):
    """Get the convective heat transfer coefficient on the gas side. 
    Uses Eqn (8-22) on page 312 or RPE 7th edition.

    Args:
        D (float): Flow diameter (m)
        M (float): Freestream Mach number
        T (float): Freestream temperature (K)
        rho (float): Freestream density (kg/m^3)
        gamma (float): Ratio of specific heats (cp/cv)
        R (float): Specific gas constant (J/kg/K)
        mu (float): Freestream absolute viscosity (Pa s)
        k (float): Freestream thermal conductivity (W/m/K)
        Pr (float): Freestream Prandtl number

    Returns:
        float: Convective heat transfer coefficient, h, for the exhaust gas side (where q = h(T - T_inf)).
    """

    v = M * (gamma*R*T)**0.5    #Gas velocity

    return 0.026 * (rho*v)**0.8 / (D**0.2) * (Pr**0.4) * k/(mu**0.8)

def h_gas_2(D, cp_inf, mu_inf, Pr_inf, rho_inf, v_inf, rho_am, mu_am, mu0):
    """Bartz equation, 
    usingEquation (8-23) from page 312 of RPE 7th edition. 'am' refers to the gas being at the 'arithmetic mean' of the wall and freestream temperatures.

    Note:
        Seems to provide questionable results - may have been implemented incorrectly.

    Args:
        D (float): Gas flow diameter (m)
        cp_inf (float): Specific heat capacity at constant pressure for the gas, in the freestream
        mu_inf (float): Absolute viscosity in the freestream
        Pr_inf (float): Prandtl number in the freestream
        rho_inf (float): Density of the gas in the freestream
        v_inf (float): Velocity of the gas in in the freestream
        rho_am (float): Density of the gas, at T = (T_wall + T_freestream)/2
        mu_am (float): Absolute viscosity of the gas, at T = (T_wall + T_freestream)/2
        mu0 (float): Absolute viscosity of the gas under stagnation conditions.

    Returns:
        float: Convective heat transfer coefficient, h, for the exhaust gas side (where q = h(T - T_inf)).
    """

    return (0.026/D**0.2) * (cp_inf*mu_inf**0.2)/(Pr_inf**0.6) * (rho_inf * v_inf)**0.8 * (rho_am/rho_inf) * (mu_am/mu0)**0.2

def h_gas_3(c_star, At, A, pc, Tc, M, Tw, mu, cp, gamma, Pr):
    """Alternative equation for Bartz heat transfer coefficient.

    Args:
        c_star (float): C* efficiency ( = pc * At / mdot)
        At (float): Throat area (m^2)
        A (float): Flow area (m^2)
        pc (float): Chamber pressure (Pa)
        Tc (float): Chamber temperature (K)
        M (float): Freestream Mach number
        Tw (float): Wall temperature (K)
        mu (float): Freestream absolute viscosity (Pa s).
        cp (float): Gas specific heat capacity (J/kg/K)
        gamma (float): Gas ratio of specific heats (cp/cv)
        Pr (float): Freestream Prandtl number

    Returns:
        float: Convective heat transfer coefficient, h, for the exhaust gas side (where q = h(T - T_inf)).
    """

    Dt = (At *4/np.pi)**0.5
    sigma = (0.5 * (Tw/Tc) * (1 + (gamma-1)/2 * M**2) + 0.5)**0.68 * (1 + (gamma-1)/2 * M**2)**(-0.12)
    
    return (0.026)/(Dt**0.2) * (mu**0.2*cp/Pr**0.6) * (pc/c_star)**0.8 * (At/A)**0.9 * sigma

def h_coolant_1(A, D, mdot, mu, k, c_bar, rho):
    """Get the convective heat transfer coefficient for the coolant side.
    Uses the equation from page 317 of RPE 7th edition.

    Args:
        A (float): Coolant flow area (m^2)
        D (float): Coolant channel effective diameter (m)
        mdot (float): Coolant mass flow rate (kg/s)
        mu (float): Coolant absolute viscosity (Pa s)
        k (float): Coolant thermal conductivity (W/m/K)
        c_bar (float): Average specific heat capacity of the coolant (J/kg/K)
        rho (float): Coolant density (kg/m^3)

    Returns:
        float: Convective heat transfer coefficient, h, for the coolant side (where q = h(T - T_inf)).
    """
    v = mdot / (rho*A)
    return 0.023*c_bar * (mdot/A) * (D*v*rho/mu)**(-0.2) * (mu*c_bar/k)**(-2/3)


class EngineGeometry:
    """Class for storing and calculating features of the engine's geometry.

    Args:
        nozzle (float): Nozzle of the engine.
        chamber_length (float): Length of the combustion chamber (m)
        chamber_area (float): Cross sectional area of the combustion chamber (m^2)
        wall_thickness (float or array): Thickness of the inner liner wall (m). Can be constant (float), or variable (array).
        geometry (str, optional): Geometry system to use. Currently the only option is 'auto'. Defaults to "auto".

    """
    def __init__(self, nozzle, chamber_length, chamber_area, wall_thickness, geometry="auto"):
        self.nozzle = nozzle
        self.chamber_length = chamber_length
        self.chamber_area = chamber_area
        self.chamber_radius = (chamber_area/np.pi)**0.5 
        if type(wall_thickness) is (float or int):
            self.wall_thickness = [wall_thickness]  #Convert into a list so the interpolation works
        else:
            self.wall_thickness = wall_thickness
        self.geometry = geometry

        if self.nozzle.At > self.chamber_area:
            raise ValueError(f"The combustion chamber area {self.chamber_area} m^2 is smaller than the throat area {self.nozzle.At} m^2.")

        if self.geometry == "auto":
            #Use the system defined in Reference [1] - mostly using Eqns (4)
            #Make sure we cap the size of the converging section to the radius of the combustion chamber.
            chamber_radius = (self.chamber_area/np.pi)**0.5
            theta_min = -np.pi - np.arcsin((chamber_radius - self.nozzle.Rt - 1.5*self.nozzle.Rt) / (1.5*self.nozzle.Rt)) 
            if theta_min > -3*np.pi/4:
                self.theta_curved_converging_start = theta_min
            else:
                self.theta_curved_converging_start = -3*np.pi/4

            #Find key properties for the converging section
            self.x_curved_converging_start = 1.5*self.nozzle.Rt*np.cos(self.theta_curved_converging_start)
            self.y_curved_converging_start = 1.5*self.nozzle.Rt*np.sin(self.theta_curved_converging_start) + 1.5*self.nozzle.Rt + self.nozzle.Rt

            #Find the gradient where the curved converging bit starts
            dxdtheta_curved_converging_start = -1.5*self.nozzle.Rt*np.sin(self.theta_curved_converging_start)
            self.dydx_curved_converging_start = -1.5*self.nozzle.Rt*np.cos(self.theta_curved_converging_start)/dxdtheta_curved_converging_start

            #Find the x-position where we reach the combustion chamber radius
            self.x_chamber_end = self.x_curved_converging_start - (self.chamber_radius - self.y_curved_converging_start)/self.dydx_curved_converging_start

            #Start and end points of the engine
            self.x_min = self.x_chamber_end - self.chamber_length
            self.x_max = self.nozzle.length

    def y(self, x):
        """Get the radius of the engine contour for a given x position.

        Args:
            x (float): x position (m). x = 0 is the throat, x > 0 is the nozzle diverging section.

        Returns:
            float: Radius of the engine contour (m).
        """
        if self.geometry == "auto":
            #Curved converging section
            if x < 0 and x > self.x_curved_converging_start:
                theta = -np.arccos(x/(1.5*self.nozzle.Rt))
                return 1.5*self.nozzle.Rt*np.sin(theta) + 1.5*self.nozzle.Rt + self.nozzle.Rt

            #Before the curved part of the converging section
            elif x <= self.x_curved_converging_start:
                #Inside the chamber
                if x < self.x_chamber_end and x >= self.x_min:
                    return self.chamber_radius

                #Inside the converging section
                elif x >= self.x_chamber_end:
                    return np.interp(x, [self.x_chamber_end, self.x_curved_converging_start], [self.chamber_radius, self.y_curved_converging_start])

                #Outside of the engine
                else:
                    return ValueError(f"x is beyond the front of the engine. You tried to input {x} but the minimum value you're allowed is {self.x_chamber_end - self.chamber_length}")
            
            #In the diverging section of the nozzle
            elif x >= 0:
                return self.nozzle.y(x)

    def A(self, x):
        """Get the engine cross sectional area at a given x position.

        Args:
            x (float): x position (m). x = 0 is the throat, x > 0 is the nozzle diverging section.

        Returns:
            float: Cross sectional area (m^2)
        """
        return np.pi*self.y(x)**2

    def plot_geometry(self, number_of_points = 1000):
        """Plots the engine geometry. Note that to see the plot, you will need to run matplotlib.pyplot.show().

        Args:
            number_of_points (int, optional): Numbers of discrete points to plot. Defaults to 1000.
        """
        x = np.linspace(self.x_min, self.x_max, number_of_points)
        y = np.zeros(len(x))

        for i in range(len(x)):
            y[i] = self.y(x[i])

        fig, axs = plt.subplots()
        axs.plot(x, y, color="blue")
        axs.plot(x, -y, color="blue")
        axs.grid()
        axs.set_aspect('equal')
        plt.xlabel("x position (m)")
        plt.ylabel("y position (m)")

class CoolingJacket:
    """Cooling jacket parameters.

    Args:
        inner_wall (Material): Inner wall material
        inlet_T (float): Inlet coolant temperature (K)
        inlet_p0 (float): Inlet coolant stagnation pressure (Pa)
        coolant_transport (TransportProperties): Container for the coolant transport properties.
        mdot_coolant (float): Coolant mass flow rate (kg/s)
        channel_shape (str, optional): Options include 'rectangle', 'semi-circle', and 'custom'. Defaults to "rectangle".
        configuration (str, optional): Options include 'spiral'. Defaults to "spiral".
    
    Keyword Args:
        rectangle_width (float, optional): If using channel_shape = 'rectangle', this is the height of the rectangles (in the radial direction).
        rectangle_height (float, optional): If using channel_shape = 'rectangle, this is the width of the rectangles (in the hoopwise direction). 
        circle_diameter (float, optional): If using channel_shape = 'semi-circle', this is the diameter of the semi circle.
        custom_effective_diameter (float, optional) : If using channel_shape = 'custom', this is the effective diameter you want to use. 
        custom_flow_area (float, optional) : If using channel_shape = 'custom', this is the flow you want to use. 
    """
    def __init__(self, inner_wall, inlet_T, inlet_p0, coolant_transport, mdot_coolant, channel_shape = "rectangle", configuration = "spiral", **kwargs):

        self.inner_wall = inner_wall
        self.coolant_transport = coolant_transport       
        self.mdot_coolant = mdot_coolant
        self.inlet_T = inlet_T
        self.inlet_p0 = inlet_p0
        self.channel_shape = channel_shape
        self.configuration = configuration
        
        if self.channel_shape == "rectangle":
            #Page 317 of RPE 7th Edition
            self.rectangle_width = kwargs["rectangle_width"]
            self.rectangle_height = kwargs["rectangle_height"]
            self.perimeter = 2*self.rectangle_width + 2*self.rectangle_height
            self.flow_area = self.rectangle_width*self.rectangle_height
            self.hydraulic_radius = self.flow_area/self.perimeter
            self.effective_diameter = 4*self.hydraulic_radius

        if self.channel_shape == "semi-circle":
            self.circle_diameter = kwargs["circle_diameter"]
            self.perimeter = self.circle_diameter + np.pi*self.circle_diameter/2
            self.flow_area = np.pi*self.circle_diameter**2/8
            self.hydraulic_radius = self.flow_area/self.perimeter
            self.effective_diameter = 4*self.hydraulic_radius

        if self.channel_shape == "custom":
            self.flow_area = kwargs["custom_flow_area"]
            self.effective_diameter = kwargs["custom_effective_diameter"]

    def A(self, x=None):
        """Get coolant channel cross flow cross sectional area.

        Args:
            x (float, optional): Axial position along the engine. This parameter may have no effect on the output. Defaults to None.

        Returns:
            float: Cooling channel cross flow area (m^2)
        """
        return self.flow_area
    
    def D(self, x=None):
        """Get the 'effective diameter' of the cooling channel. This is equal 4*hydraulic_radius, with hydraulic_radius = channel_area / channel_perimeter.

        Args:
            x (float, optional): Axial position along the engine. This parameter may have no effect on the output. Defaults to None.

        Returns:
            float: Effective diameter (m)
        """
        return self.effective_diameter

    def p0(self, x=None):
        """Get coolant stagnation pressure as a function of position (currently implemented as constant)

        Args:
            x (float, optional): x position, throat at x = 0, nozzle at x > 0. Defaults to None.

        Returns:
            float: Coolant stagnation pressure
        """
        return self.inlet_p0

class Material:
    """Class used to specify a material and its properties. 
    Used specifically for defining the inner liner of an engine.

    Args:
        E (float): Young's modulus (Pa)
        sigma_y (float): 0.2% yield stress (Pa)
        poisson (float): Poisson's ratio
        alpha (float): Thermal expansion coefficient (strain/K)
        k (float): Thermal conductivity (W/m/K)

    """
    def __init__(self, E, sigma_y, poisson, alpha, k):
        self.E = E                  
        self.sigma_y = sigma_y      
        self.poisson = poisson      
        self.alpha = alpha          
        self.k = k                  

        self.perf_therm = (1 - self.poisson) * self.k / (self.alpha * self.E)   #Performance coefficient for thermal stress, higher is better

class TransportProperties:
    """Container for transport properties of a fluid.

    Args:
        model (str, optional): The module to use for modelling. Intended to offer 'thermo', 'CoolProp' and 'cantera', but only thermo works as of now. Defaults to "thermo".
        
    Keywords Args:
        thermo_object (thermo.chemical.Chemical or thermo.mixture.Mixture): An object from the 'thermo' Python module.
        coolprop_name (str): Name of the chemcial or mixture for the CoolProp module. See http://www.coolprop.org/ for a list of available fluids.
    """

    def __init__(self, model = "thermo", **kwargs):
        self.model = model

        if model == "thermo":
            self.thermo_object = kwargs["thermo_object"]

        elif model == "CoolProp":
            self.coolprop_name = kwargs["coolprop_name"]
        
        else:
            raise ValueError(f"The model {model} is not a valid option.")

    def check_liquid(self, T, p):
        """Returns True if the fluid is a liquid at the given temperature and pressure. Used to check for coolant boil-off.

        Args:
            T (float): Temperature (K)
            p (float): Pressure (Pa)

        Returns:
            bool: True if the fluid is liquid, False if it's any other phase
        """
        if self.model == "thermo":
            self.thermo_object.calculate(T = T, P = p) 
            if self.thermo_object.phase == 'l':
                return True
            else:
                return False
            
        elif self.model == "CoolProp":
            #CoolProp uses a phase index of '0' to refer to the liquid state
            if PropsSI("PHASE", "T", T, "P", p, self.coolprop_name) == 0:
                return True
            else:
                return False

    def k(self, T, p):
        """Thermal conductivity

        Args:
            T (float): Temperature (K)
            p (float): Pressure (Pa)

        Returns:
            float: Thermal conductivity
        """
        if self.model == "thermo":
            self.thermo_object.calculate(T = T, P = p) 
            return self.thermo_object.k
            
        elif self.model == "CoolProp":
            return PropsSI("CONDUCTIVITY", "T", T, "P", p, self.coolprop_name)

    def mu(self, T, p):
        """Absolute viscosity

        Args:
            T (float): Temperature (K)
            p (float): Pressure (Pa)

        Returns:
            float: Absolute viscosity
        """
        if self.model == "thermo":
            self.thermo_object.calculate(T = T, P = p) 
            return self.thermo_object.mu

        elif self.model == "CoolProp":
            return PropsSI("VISCOSITY", "T", T, "P", p, self.coolprop_name)

    def Pr(self, T, p):
        """Prandtl number

        Args:
            T (float): Temperature (K)
            p (float): Pressure (Pa)

        Returns:
            float: Prandtl number
        """
        if self.model == "thermo":
            self.thermo_object.calculate(T = T, P = p) 
            return self.thermo_object.Pr

        elif self.model == "CoolProp":
            return PropsSI("PRANDTL", "T", T, "P", p, self.coolprop_name)

    def cp(self, T, p):
        """Specific heat capacity at constant pressure (J/kg/K)

        Args:
            T (float): Temperature (K)
            p (float): Pressure (Pa)

        Returns:
            float: Specific heat capacity at constant pressure (J/kg/K)

        """
        if self.model == "thermo":
            self.thermo_object.calculate(T = T, P = p) 
            return self.thermo_object.Cp
        
        elif self.model == "CoolProp":
            return PropsSI("CPMASS", "T", T, "P", p, self.coolprop_name)

    def rho(self, T, p):
        """Density (kg/m3)

        Args:
            T (float): Temperature (K)
            p (float): Pressure (Pa)

        Returns:
            float: Density (kg/m3)

        """
        if self.model == "thermo":
            self.thermo_object.calculate(T = T, P = p) 
            return self.thermo_object.rho
        
        elif self.model == "CoolProp":
            return PropsSI("DMASS", "T", T, "P", p, self.coolprop_name)

class EngineWithCooling:
    """Used for running cooling system analyses.

    Args:
        chamber_conditions (ChamberConditions): Engine chamber conditions object.
        geometry (EngineGeometry): Engine geometry.
        cooling_jacket (CoolingJacket): Cooling jacket properties.
        perfect_gas (PerfectGas): Properties of the exhaust gas.
        exhaust_transport (TransportProperties): Container for the exhaust gas transport properties.
    """
    def __init__(self, chamber_conditions, geometry, cooling_jacket, perfect_gas, exhaust_transport):
        self.chamber_conditions = chamber_conditions
        self.geometry = geometry
        self.cooling_jacket = cooling_jacket
        self.perfect_gas = perfect_gas
        self.exhaust_transport = exhaust_transport

        self.c_star = self.chamber_conditions.p0 * self.geometry.nozzle.At / self.chamber_conditions.mdot

    def M(self, x):
        """Get exhaust gas Mach number.

        Args:
            x (float): Axial position along the engine (m). Throat is at x = 0.

        Returns:
            float: Mach number of the freestream.
        """
        #If we're at the throat M=1 by default:
        if x==0:
            return 1.00

        #If we're not at the throat:
        else:
            def func_to_solve(Mach):
                return self.chamber_conditions.mdot*(self.perfect_gas.cp*self.chamber_conditions.T0)**0.5 / (self.geometry.A(x)*self.chamber_conditions.p0) - bam.m_bar(Mach, self.perfect_gas.gamma)
            
            if x > 0:
                Mach = scipy.optimize.root_scalar(func_to_solve, bracket = [1,300], x0 = 1).root
            else:
                Mach = scipy.optimize.root_scalar(func_to_solve, bracket = [0,1], x0 = 0.5).root

            return Mach

    def T(self, x):
        """Get exhaust gas temperature.

        Args:
            x (float): Axial position (m). Throat is at x = 0.

        Returns:
            float: Temperature (K)
        """
        return bam.T(self.chamber_conditions.T0, self.M(x), self.perfect_gas.gamma)

    def p(self, x):
        """Get exhaust gas pressure.

        Args:
            x (float): Axial position (m). Throat is at x = 0.

        Returns:
            float: Freestream pressure (Pa)
        """
        return bam.p(self.chamber_conditions.p0, self.M(x), self.perfect_gas.gamma)

    def rho(self, x):
        """Get exhaust gas density.

        Args:
            x (float): Axial position. Throat is at x = 0.

        Returns:
            float: Freestream gas density (kg/m^3)
        """
        #p = rhoRT for an ideal gas, so rho = p/RT
        return self.p(x)/(self.T(x)*self.perfect_gas.R)

    def show_gas_temperature(self, number_of_points=1000):
        """Plot freestream gas temperature against position. Note that to see the plot, you will need to run matplotlib.pyplot.show().

        Args:
            number_of_points (int, optional): Number of points to discretise the plot into. Defaults to 1000.
        """
        x = np.linspace(self.geometry.x_min, self.geometry.x_max, number_of_points)
        y = np.zeros(len(x))
        T = np.zeros(len(x))

        for i in range(len(x)):
            y[i] = self.geometry.y(x[i])
            T[i] = self.T(x[i])

        fig, ax_shape = plt.subplots()

        ax_shape.plot(x, y, color="blue")
        ax_shape.plot(x, -y, color="blue")
        ax_shape.set_aspect('equal')
        ax_shape.set_xlabel("x position (m)")
        ax_shape.set_ylabel("y position (m)")

        ax_temp = ax_shape.twinx()
        ax_temp.plot(x, T, color="orange")
        ax_temp.grid()
        ax_temp.set_ylabel("Temperature (K)")

    def show_gas_mach(self, number_of_points=1000):
        """Plot Mach number against position. Note that to see the plot, you will need to run matplotlib.pyplot.show().

        Args:
            number_of_points (int, optional): Number of points to discretise the plot into. Defaults to 1000.
        """
        x = np.linspace(self.geometry.x_min, self.geometry.x_max, number_of_points)
        y = np.zeros(len(x))
        M = np.zeros(len(x))

        for i in range(len(x)):
            y[i] = self.geometry.y(x[i])
            M[i] = self.M(x[i])

        fig, ax_shape = plt.subplots()

        ax_shape.plot(x, y, color="blue")
        ax_shape.plot(x, -y, color="blue")
        ax_shape.set_aspect('equal')
        ax_shape.set_xlabel("x position (m)")
        ax_shape.set_ylabel("y position (m)")

        ax_temp = ax_shape.twinx()
        ax_temp.plot(x, M, color="green")
        ax_temp.grid()
        ax_temp.set_ylabel("Mach number")

    def coolant_velocity(self, x, rho_coolant):
        """Get coolant velocity

        Args:
            x (float): Axial position
            rho_coolant (float): Coolant density (kg/m3)

        Returns:
            float: Coolant velocity (m/s)
        """
        return self.cooling_jacket.mdot_coolant/(rho_coolant * self.cooling_jacket.A(x))

    def map_liner_profile(self, number_of_points=1000):
        """Maps the provided liner thickness profile to the engine geometry,
           so each element in cooling analysis has a thickness value.
           
           Args:
                number_of_points (int): Number of discrete liner positions

           Returns:
                liner (array): Interpolated liner thickness profile
           """
        liner = np.zeros(number_of_points)
        for i in range(number_of_points):
            x_pos = i*self.geometry.chamber_length/number_of_points
            # How far along the engine is the current point
            liner_index = x_pos * len(self.geometry.wall_thickness)
            liner[i] = np.interp(liner_index, range(len(self.geometry.wall_thickness)), self.geometry.wall_thickness)
        return liner

    def thermal_circuit(self, x, h_gas, h_coolant, inner_wall, wall_thickness, T_gas, T_coolant):
        """
        q is per unit length along the nozzle wall (axially) - positive when heat is flowing to the coolant.  
        q_Adot is the heat flux per unit area along the nozzle wall.  
        Uses the idea of thermal circuits and resistances - we have three resistors in series.

        Args:
            x (float): x position (m)
            h_gas (float): Gas side convective heat transfer coefficient
            h_coolant (float): Coolant side convective heat transfer coefficient
            inner_wall (Material): Material object for the inner wall, needed for thermal conductivity
            wall_thickness (float): Thickness of the inner wall at x position (m)
            T_gas (float): Free stream gas temperature (K)
            T_coolant (float): Coolant temperature (K)

        Returns:
            float, float, float, float, float: q_dot, R_gas, R_wall, R_coolant, q_Adot
        """

        r = self.geometry.y(x)
        
        r_out = r + wall_thickness
        r_in = r 

        A_in = 2*np.pi*r_out    #Inner area per unit length (i.e. just the inner circumference)
        A_out = 2*np.pi*r_in    #Outer area per unit length (i.e. just the outer circumference)

        R_gas = 1/(h_gas*A_in)
        R_wall = np.log(r_out/r_in)/(2*np.pi*inner_wall.k)
        R_coolant = 1/(h_coolant*A_out)

        q_dot = (T_gas - T_coolant)/(R_gas + R_wall + R_coolant)    #Heat flux per unit length
        q_Adot = q_dot / A_in                                       #Heat flux per unit area

        return q_dot, R_gas, R_wall, R_coolant, q_Adot

    def run_heating_analysis(self, number_of_points=1000, h_gas_model = "1", h_coolant_model = "1", to_json = "heating_output.json"):
        """Run a simulation of the engine cooling system to get wall temperatures, coolant temperatures, etc.

        Args:
            number_of_points (int, optional): Number of discrete points to divide the engine into. Defaults to 1000.
            h_gas_model (str, optional): Equation to use for the gas side convective heat transfer coefficients. Options are '1', '2' and '3'. Defaults to "1".
            h_coolant_model (str, optional): Equation to use for the coolant side convective heat transfer coefficients, currently the only option is '1'. Defaults to "1".
            to_json (str or bool, optional): Directory to export a .JSON file to, containing simulation results. If False, no .JSON file is saved. Defaults to 'heating_output.json'.

        Returns:
            dict: Results of the simulation. Contains the following dictionary keys: 
                - "x" : x positions corresponding to the rest of the data (m)
                - "T_wall_inner" : Exhaust gas side wall temperature (K)
                - "T_wall_outer" : Coolant side wall temperature (K)
                - "T_coolant" : Coolant temperature (K)
                - "T_gas" : Exhaust gas freestream temperature (K)
                - "q_dot" : Heat transfer rate per unit length (axially along the engine) (W/m)
                - "q_Adot": Heat transfer rate per unit area (W/m^2)
                - "h_gas" : Convective heat transfer rate for the exhaust gas side
                - "h_coolant" : Convective heat transfer rate for the coolant side
                - "boil_off_position" : x position of any coolant boil off. Equal to None if the coolant does not boil.
        """

        '''Initialise variables and arrays'''
        #To keep track of any coolant boiling
        boil_off_position = None
        
        #Discretisation of the nozzle
        discretised_x = np.linspace(self.geometry.x_max, self.geometry.x_min, number_of_points) #Run from the back end (the nozzle exit) to the front (chamber)
        dx = discretised_x[0] - discretised_x[1]

        #Discretised liner thickness
        liner = self.map_liner_profile(number_of_points)

        #Temperatures and heat transfer rates
        T_wall_inner = np.zeros(len(discretised_x)) #Gas side wall temperature
        T_wall_outer = np.zeros(len(discretised_x)) #Coolant side wall temperature
        T_coolant = np.zeros(len(discretised_x))    #Coolant temperature
        T_gas = np.zeros(len(discretised_x))        #Freestream gas temperature
        q_dot = np.zeros(len(discretised_x))        #Heat transfer rate per unit length
        q_Adot = np.zeros(len(discretised_x))       #Heat transfer rate per unit area
        h_gas = np.zeros(len(discretised_x))
        h_coolant = np.zeros(len(discretised_x))

        '''Main loop'''
        for i in range(len(discretised_x)):
            x = discretised_x[i]
            T_gas[i] = self.T(x)

            #Calculate the current coolant temperature
            if i == 0:
                T_coolant[i] = self.cooling_jacket.inlet_T

            else:
                #Increase in coolant temperature, q*dx = mdot*Cp*dT
                T_coolant[i] = T_coolant[i-1] + (q_dot[i-1]*dx)/(self.cooling_jacket.mdot_coolant*cp_coolant)   

            #Update coolant heat capacity
            cp_coolant = self.cooling_jacket.coolant_transport.cp(T = T_coolant[i], p = self.cooling_jacket.p0(x))

            #Gas side heat transfer coefficient
            if h_gas_model == "1":
                h_gas[i] = h_gas_1(2*self.geometry.y(x),
                                   self.M(x),
                                   T_gas[i],
                                   self.rho(x),
                                   self.perfect_gas.gamma,
                                   self.perfect_gas.R,
                                   self.exhaust_transport.mu(T = T_gas[i], p = self.p(x)),
                                   self.exhaust_transport.k(T = T_gas[i], p = self.p(x)),
                                   self.exhaust_transport.Pr(T = T_gas[i], p = self.p(x)))

            elif h_gas_model == "2":
                gamma = self.perfect_gas.gamma
                R = self.perfect_gas.R
                D = 2*self.geometry.y(x)            #Flow diameter

                #Freestream properties
                p_inf = self.p(x)
                T_inf = T_gas[i]
                rho_inf = self.rho(x)
                M_inf = self.M(x)
                v_inf = M_inf * (gamma*R*T_inf)**0.5    #Gas velocity
                mu_inf = self.exhaust_transport.mu(T = T_gas[i], p = p_inf)
                Pr_inf = self.exhaust_transport.Pr(T = T_gas[i], p = p_inf)
                cp_inf = self.perfect_gas.cp

                #Properties at arithmetic mean of T_wall and T_inf
                T_am = (T_inf + T_wall_inner[i-1]) / 2
                mu_am = self.exhaust_transport.mu(T = T_am, p = p_inf)
                rho_am = p_inf/(R*T_am)                                 #p = rho R T - pressure is roughly uniform across the boundary layer so p_inf ~= p_wall

                #Stagnation properties
                p0 = self.chamber_conditions.p0
                T0 = self.chamber_conditions.T0
                mu0 = self.exhaust_transport.mu(T =  T0, p = p0)

                h_gas[i] = h_gas_2(D, cp_inf, mu_inf, Pr_inf, rho_inf, v_inf, rho_am, mu_am, mu0)

            elif h_gas_model == "3":
                h_gas[i] = h_gas_3(self.c_star,
                                   self.geometry.nozzle.At, 
                                   self.geometry.A(x), 
                                   self.chamber_conditions.p0, 
                                   self.chamber_conditions.T0, 
                                   self.M(x), 
                                   T_wall_inner[i-1], 
                                   self.exhaust_transport.mu(T = T_gas[i], p = self.p(x)), 
                                   self.perfect_gas.cp, 
                                   self.perfect_gas.gamma, 
                                   self.exhaust_transport.Pr(T = T_gas[i], p = self.p(x)))

            else:
                raise AttributeError(f"Could not find the h_gas_model '{h_gas_model}'")
            
            #Coolant side heat transfer coefficient
            if h_coolant_model == "1":
                h_coolant[i] = h_coolant_1(self.cooling_jacket.A(x), 
                                           self.cooling_jacket.D(x), 
                                           self.cooling_jacket.mdot_coolant, 
                                           self.cooling_jacket.coolant_transport.mu(T = T_coolant[i], p = self.cooling_jacket.p0(x)), 
                                           self.cooling_jacket.coolant_transport.k(T = T_coolant[i], p = self.cooling_jacket.p0(x)), 
                                           cp_coolant, 
                                           self.cooling_jacket.coolant_transport.rho(T = T_coolant[i], p = self.cooling_jacket.p0(x)))

            else:
                raise AttributeError(f"Could not find the h_coolant_model '{h_coolant_model}'")
            
            #Check for coolant boil off - a CoolProp uses a phase index of '0' to refer to the liquid state (see http://www.coolprop.org/coolprop/HighLevelAPI.html)

            if boil_off_position == None and self.cooling_jacket.coolant_transport.check_liquid(T = T_coolant[i], p = self.cooling_jacket.p0(x)) == False:
                print(f"WARNING: Coolant boiled off at x = {x} m")
                boil_off_position = x

            #Get thermal circuit properties
            q_dot[i], R_gas, R_wall, R_coolant, q_Adot[i] = self.thermal_circuit(x, h_gas[i], h_coolant[i], self.cooling_jacket.inner_wall, liner[i], T_gas[i], T_coolant[i])

            #Calculate wall temperatures
            T_wall_inner[i] = T_gas[i] - q_dot[i]*R_gas
            T_wall_outer[i] = T_wall_inner[i] - q_dot[i]*R_wall

        #Dictionary containing results
        output_dict = {"x" : list(discretised_x),
                "T_wall_inner" : list(T_wall_inner),
                "T_wall_outer" : list(T_wall_outer),
                "T_coolant" : list(T_coolant),
                "T_gas" : list(T_gas),
                "q_dot" : list(q_dot),
                "q_Adot": list(q_Adot),
                "h_gas" : list(h_gas),
                "h_coolant" : list(h_coolant),
                "boil_off_position" : boil_off_position}

        #Export a .JSON file if required
        if to_json != False:
            with open(to_json, "w+") as write_file:
                json.dump(output_dict, write_file)
                print("Exported JSON data to '{}'".format(to_json))

        return output_dict

    def run_stress_analysis(self, heating_result, type="thermal", condition="steady"):
        """Perform stress analysis on the liner, using a cooling result.
        Args:
            heating_result (dict): Requires a heating analysis result to compute stress.
            type (str, optional): Options are "pressure",  "thermal" and "combined". Defaults to "thermal". (ONLY DEFAULT WORKS)
            condition (str, optional): Engine state for analysis. Options are "steady", "startup", or "shutdown". Defaults to "steady". (ONLY DEFAULT WORKS)

        Returns:
            dict: Analysis result. 'thermal_stress' is the heat induced stress, 'deltaT_wall' is the wall temperature difference, hot side - cold side
        """
        length = len(heating_result["x"])
        wall_stress = np.zeros(length)
        wall_deltaT = np.zeros(length)

        material = self.cooling_jacket.inner_wall

        for i in range(length):
            wall_deltaT[i] = heating_result["T_wall_inner"][i] - \
                heating_result["T_wall_outer"][i]

        # Compute wall temperature gradient
        wall_deltaT = wall_deltaT[::-1]
        # Makes the data order match with the coolant flow direction, i.e. nozzle exit to injector face
        # Spent an hour wondering why the throat was cooler than the chamber wall...

        for i in range(length):
            cur_stress = material.k * \
                wall_deltaT[i] / (2 * material.perf_therm)
        # Determine thermal stress using Ref [3], P53:
        # sigma_thermal = E*alpha*q_w*deltaL/(2*(1-v)k_w) =
        # E*alpha*deltaT/2(1-v)

            wall_stress[i] = cur_stress

        return {"thermal_stress": wall_stress,
                "deltaT_wall": wall_deltaT}