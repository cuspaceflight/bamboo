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
import scipy


#Check if CoolProp is installed
import imp
try:
    imp.find_module('CoolProp')
    CoolProp_available = True
    from CoolProp.CoolProp import PropsSI

except ImportError:
    CoolProp_available = False

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


class Material:
    """Class used to specify a material and its properties. 
    Used specifically for defining the inner liner of an engine.

    Args:
        E (float): Young's modulus (Pa)
        sigma_y (float): 0.2% yield stress (Pa)
        poisson (float): Poisson's ratio
        alpha (float): Thermal expansion coefficient (strain/K)
        k (float): Thermal conductivity (W/m/K)

    Keyword Args:
        c (float): Specific heat capacity (J/kg/K)
        rho (float): Density (kg/m^3)
    """
    def __init__(self, E, sigma_y, poisson, alpha, k, **kwargs):
        self.E = E                  
        self.sigma_y = sigma_y      
        self.poisson = poisson      
        self.alpha = alpha          
        self.k = k                  

        if "c" in kwargs:
            self.c = kwargs["c"]
        if "rho" in kwargs:
            self.rho = kwargs["rho"]

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
            if CoolProp_available:
                self.coolprop_name = kwargs["coolprop_name"]
            else:
                raise ImportError("Could not find the 'CoolProp' module, so can't use TransportProperties.model = 'CoolProp'")
        
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

class CoolingJacket:
    """Container for cooling jacket information - e.g. for regenerative cooling.

    Args:
        inner_wall (Material): Inner wall material
        inlet_T (float): Inlet coolant temperature (K)
        inlet_p0 (float): Inlet coolant stagnation pressure (Pa)
        coolant_transport (TransportProperties): Container for the coolant transport properties.
        mdot_coolant (float): Coolant mass flow rate (kg/s)
        xs (list): x position that the cooling jacket starts and ends at in the form [x_start, x_end]. Defaults to [-1000, 1000].
        configuration (str, optional): Options include 'spiral' and 'vertical'. Defaults to "vertical".
    
    Keyword Args:
        channel_shape (str, optional): Used if configuration = 'spiral'. Options include 'rectangle', 'semi-circle', and 'custom'. 
        channel_height (float, optional): If using configuration = 'vertical' or channel_shape = 'rectangle', this is the height of the channels (m).
        channel_width (float, optional): If using channel_shape = 'rectangle', this is the width of the channels (m).
        channel_diameter (float, optional): If using channel_shape = 'semi-circle', this is the diameter of the semi circle.
        custom_effective_diameter (float, optional): If using channel_shape = 'custom', this is the effective diameter you want to use. 
        custom_flow_area (float, optional): If using channel_shape = 'custom', this is the flow you want to use. 
    """
    def __init__(self, inner_wall, inlet_T, inlet_p0, coolant_transport, mdot_coolant, xs = [-1000, 1000], configuration = "spiral", **kwargs):

        self.inner_wall = inner_wall
        self.coolant_transport = coolant_transport       
        self.mdot_coolant = mdot_coolant
        self.xs = xs
        self.inlet_T = inlet_T
        self.inlet_p0 = inlet_p0
        self.configuration = configuration
        
        if self.configuration == 'spiral':
            self.channel_shape = kwargs['channel_shape']

            if self.channel_shape == "rectangle":
                #Page 317 of RPE 7th Edition
                self.channel_width = kwargs["rectangle_width"]
                self.channel_height = kwargs["channel_height"]
                self.perimeter = 2*self.channel_width + 2*self.channel_height
                self.flow_area = self.channel_width*self.channel_height
                self.hydraulic_radius = self.flow_area/self.perimeter
                self.effective_diameter = 4*self.hydraulic_radius

            if self.channel_shape == "semi-circle":
                self.channel_diameter = kwargs["channel_diameter"]
                self.perimeter = self.channel_diameter + np.pi*self.channel_diameter/2
                self.flow_area = np.pi*self.channel_diameter**2/8
                self.hydraulic_radius = self.flow_area/self.perimeter
                self.effective_diameter = 4*self.hydraulic_radius

            if self.channel_shape == "custom":
                self.flow_area = kwargs["custom_flow_area"]
                self.effective_diameter = kwargs["custom_effective_diameter"]
        
        elif self.configuration == 'vertical':
            self.channel_height = kwargs["channel_height"]

    def A(self, x=None):
        """Get coolant channel cross flow cross sectional area.

        Args:
            x (float, optional): Axial position along the engine. This parameter may have no effect on the output. Defaults to None.

        Returns:
            float: Cooling channel cross flow area (m^2)
        """
        if self.configuration == 'spiral':
            return self.flow_area

        elif self.configuration == 'vertical':
            raise AttributeError("The area for 'vertical' style cooling channels depends on the engine geometry, so must be calculated manually.")
    
        else:
            raise ValueError(f"The cooling jacket configuration {self.configuration} is not recognised. Try 'spiral' or 'vertical'. ")

    def D(self, x=None):
        """Get the 'effective diameter' of the cooling channel. This is equal 4*hydraulic_radius, with hydraulic_radius = channel_area / channel_perimeter.

        Args:
            x (float, optional): Axial position along the engine. This parameter may have no effect on the output. Defaults to None.

        Returns:
            float: Effective diameter (m)
        """
        if self.configuration == 'spiral':
            return self.effective_diameter

        elif self.configuration == 'vertical':
            raise AttributeError("The effective diameter for 'vertical' style cooling channels depends on the engine geometry, so must be calculated manually.")

        else:
            raise ValueError(f"The cooling jacket configuration {self.configuration} is not recognised. Try 'spiral' or 'vertical'. ")

    def p0(self, x=None):
        """Get coolant stagnation pressure as a function of position (currently implemented as constant)

        Args:
            x (float, optional): x position, throat at x = 0, nozzle at x > 0. Defaults to None.

        Returns:
            float: Coolant stagnation pressure
        """
        return self.inlet_p0

    def coolant_velocity(self, x, rho_coolant):
        """Get coolant velocity using mdot = rho*V*A.

        Args:
            x (float): Axial position
            rho_coolant (float): Coolant density (kg/m3)

        Returns:
            float: Coolant velocity (m/s)
        """
        if self.configuration == 'spiral':
            return self.mdot_coolant/(rho_coolant * self.A(x))

        elif self.configuration == 'vertical':
            raise AttributeError("The coolant velocity for 'vertical' style cooling channels depends on the engine geometry, so must be calculated manually.")
    
        else:
            raise ValueError(f"The cooling jacket configuration {self.configuration} is not recognised. Try 'spiral' or 'vertical'. ")

class Ablative:
    def __init__(self, ablative_material, wall_material, wall_thickness, regression_rate, xs = [-1000, 1000]):
        self.ablative_material = ablative_material
        self.wall_material = wall_material
        self.wall_thickness = wall_thickness
        self.regression_rate = regression_rate
        self.xs = xs