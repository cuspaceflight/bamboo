'''
Extra tools for modelling the cooling system of a liquid rocket engine.

Room for improvement:
    - Run some more thorough tests on the different h_gas() methods to try and validate/compare them.
    - The EngineWithCooling.rho() function calculates rho by doing p/RT, but it might be faster to just use isentropic compressible flow relations.

Useful:
    - List of CoolProp properties: http://www.coolprop.org/coolprop/HighLevelAPI.html#table-of-string-inputs-to-propssi-function
    
References:
    - [1] - The Thrust Optimised Parabolic nozzle, AspireSpace, http://www.aspirespace.org.uk/downloads/Thrust%20optimised%20parabolic%20nozzle.pdf   \n
    - [2] - Rocket Propulsion Elements, 7th Edition  \n
    - [3] - Design and analysis of contour bell nozzle and comparison with dual bell nozzle https://core.ac.uk/download/pdf/154060575.pdf 
    - [4] - Modelling ablative and regenerative cooling systems for an ethylene/ethane/nitrous oxide liquid fuel rocket engine, Elizabeth C. Browne, https://mountainscholar.org/bitstream/handle/10217/212046/Browne_colostate_0053N_16196.pdf?sequence=1&isAllowed=y  \n
    - [5] - Thermofluids databook, CUED, http://www-mdp.eng.cam.ac.uk/web/library/enginfo/cueddatabooks/thermofluids.pdf    \n
    - [6] - Comparison of empirical correlations for the estimation of conjugate heat transfer in a thrust chamber, http://www.lifesciencesite.com/lsj/life0904/111_11626life0904_708_716.pdf
    - [7] - Regenerative cooling of liquid rocket engine thrust chambers, ASI, https://www.researchgate.net/profile/Marco-Pizzarelli/publication/321314974_Regenerative_cooling_of_liquid_rocket_engine_thrust_chambers/links/5e5ecd824585152ce804e244/Regenerative-cooling-of-liquid-rocket-engine-thrust-chambers.pdf  \n
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
    """Get the convective heat transfer coefficient on the gas side, non-Bartz equation.
    Uses Eqn (8-22) on page 312 or RPE 7th edition (Reference [2])

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
    using Equation (8-23) from page 312 of RPE 7th edition (Reference [2]). 'am' refers to the gas being at the 'arithmetic mean' of the wall and freestream temperatures.

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
    """Alternative equation for Bartz heat transfer coefficient, from page 710 of Reference [6].

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
    Uses the equation from page 317 of RPE 7th edition (Reference [2]).

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

def h_coolant_2(rho, V, D, mu_bulk, mu_wall, Pr, k):
    """Sieder-Tate equation for convective heat transfer coefficient.

    Args:
        rho (float): Coolant bulk density (kg/m^3)
        V (float): Coolant bulk velocity (m/s)
        D (float): Effective diameter of pipe (m)
        mu_bulk (float): Absolute viscosity of the coolant at the bulk temperature (Pa s).
        mu_wall (float): Absolute viscosity of the coolant at the wall temperature (Pa s).
        Pr (float): Bulk Prandtl number of the coolant.
        k (float): Bulk thermal conductivity of the coolant.
    
    Returns:
        float: Convective heat transfer coefficient
    """
    Re = rho*V*D/mu_bulk
    Nu = 0.027*Re**(4/5)*Pr**(1/3)*(mu_bulk/mu_wall)**0.14

    return Nu*k/D

def h_coolant_3(rho, V, D, mu, Pr, k):
    """Dittus-Boelter equation for convective heat transfer coefficient.

    Args:
        rho (float): Coolant bulk density (kg/m^3).
        V (float): Coolant bulk velocity (m/s)
        D (float): Effective diameter of pipe (m)
        mu (float): Coolant bulk viscosity (Pa s)
        Pr (float): Coolant bulk Prandtl number
        k (float): Coolant thermal conductivity

    Returns:
        float: Convective heat transfer coefficient
    """
    Re = rho*V*D/mu
    Nu = 0.023*Re**(4/5)*Pr**0.4

    return Nu*k/D


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
    
    Attributes:
        E (float): Young's modulus (Pa)
        sigma_y (float): 0.2% yield stress (Pa)
        poisson (float): Poisson's ratio
        alpha (float): Thermal expansion coefficient (strain/K)
        k (float): Thermal conductivity (W/m/K)
        c (float): Specific heat capacity (J/kg/K). Only available if assigned.
        rho (float): Density (kg/m^3). Only available if assigned.
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

    def __repr__(self):
        return f"""bamboo.cooling.Material Object \nYoung's modulus = {self.E/1e9} GPa 
0.2% Yield Stress = {self.sigma_y/1e6} MPa 
Poisson's ratio = {self.poisson}
alpha = {self.alpha} strain/K
Thermal conductivity = {self.k} W/m/K
(may also have a specific heat capacity (self.c) and density (self.rho))"""

class TransportProperties:
    """Container for transport properties of a fluid. 
    
    Note:
        Sometimes 'thermo' uses a questionable choice of phase when calculating transport properties for mixtures (and sometimes pure chemicals), 
        If you are getting questionable data, it may be useful to try out the 'force_phase' argument.

    Args:
        model (str, optional): The module to use for modelling. Options include 'thermo' and 'CoolProp'.
        force_phase (str, optional): 'l' for liquid or 'g' for gas. Forces thermo to use transport properties in the given phase. Does not affect other models. Defaults to None.
    
    Keywords Args:
        thermo_object (thermo.chemical.Chemical or thermo.mixture.Mixture): An object from the 'thermo' Python module.
        coolprop_name (str): Name of the chemcial or mixture for the CoolProp module. See http://www.coolprop.org/ for a list of available fluids.
    """

    def __init__(self, model = "thermo", force_phase = None, **kwargs):
        self.model = model
        self.force_phase = force_phase

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

            if self.force_phase == 'l':
                return self.thermo_object.kl
            elif self.force_phase == 'g':
                return self.thermo_object.kg
            else:
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

            if self.force_phase == 'l':
                return self.thermo_object.mul
            elif self.force_phase == 'g':
                return self.thermo_object.mug
            else:
                #Manually check which phase we're in, and return the right viscosity (otherwise sometimes it seems to return odd results)
                if self.thermo_object.phase == 'g':
                    return self.thermo_object.mug
                elif self.thermo_object.phase == 'l':
                    return self.thermo_object.mul
                else:
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

            if self.force_phase == 'l':
                return self.thermo_object.Prl
            elif self.force_phase == 'g':
                return self.thermo_object.Prg
            else:
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

            if self.force_phase == 'l':
                return self.thermo_object.Cpl
            elif self.force_phase == 'g':
                return self.thermo_object.Cpg
            else:
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

            if self.force_phase == 'l':
                return self.thermo_object.rhol
            elif self.force_phase == 'g':
                return self.thermo_object.rhog
            else:
                return self.thermo_object.rho
        
        elif self.model == "CoolProp":
            return PropsSI("DMASS", "T", T, "P", p, self.coolprop_name)


class CoolingJacket:
    """Container for cooling jacket information - e.g. for regenerative cooling.

    Args:
        inner_wall (Material): Wall material on the inner side of the cooling jacket.
        inlet_T (float): Inlet coolant temperature (K)
        inlet_p0 (float): Inlet coolant stagnation pressure (Pa)
        coolant_transport (TransportProperties): Container for the coolant transport properties.
        mdot_coolant (float): Coolant mass flow rate (kg/s)
        xs (list): x positions that the cooling jacket starts and ends at, [x_min, x_max]. Defaults to [-1000, 1000].
        configuration (str, optional): Options include 'spiral' and 'vertical'. Defaults to "vertical".
    
    Keyword Args:
        channel_shape (str, optional): Used if configuration = 'spiral'. Options include 'rectangle', 'semi-circle', and 'custom'. 
        channel_height (float, optional): If using configuration = 'vertical' or channel_shape = 'rectangle', this is the height of the channels (m).
        channel_width (float, optional): If using channel_shape = 'rectangle', this is the width of the channels (m). If using channel_shape = 'semi-circle', this is the diameter of the semi circle (m).
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
                self.channel_width = kwargs["channel_width"]
                self.channel_height = kwargs["channel_height"]
                self.perimeter = 2*self.channel_width + 2*self.channel_height
                self.flow_area = self.channel_width*self.channel_height
                self.hydraulic_radius = self.flow_area/self.perimeter
                self.effective_diameter = 4*self.hydraulic_radius

            if self.channel_shape == "semi-circle":
                self.channel_width = kwargs["channel_width"]
                self.perimeter = self.channel_width + np.pi*self.channel_width/2
                self.flow_area = np.pi*self.channel_width**2/8
                self.hydraulic_radius = self.flow_area/self.perimeter
                self.effective_diameter = 4*self.hydraulic_radius

            if self.channel_shape == "custom":
                self.flow_area = kwargs["custom_flow_area"]
                self.effective_diameter = kwargs["custom_effective_diameter"]
        
        elif self.configuration == 'vertical':
            self.channel_height = kwargs["channel_height"]

    def A(self, x = None, y = None):
        """Get coolant channel cross flow cross sectional area.

        Args:
            x (float, optional): x position - does not currently affect anything.
            y (float, optional): The radius of the engine (m) (NOT the radius of the cooling channel).  Only required for 'vertical' channels. 

        Returns:
            float: Cooling channel flow area (m^2)
        """
        if self.configuration == 'spiral':
            return self.flow_area

        elif self.configuration == 'vertical':
            return np.pi*((y + self.channel_height)**2 - y**2)
    
        else:
            raise ValueError(f"The cooling jacket configuration {self.configuration} is not recognised. Try 'spiral' or 'vertical'. ")

    def D(self, x = None, y = None):
        """Get the 'effective diameter' of the cooling channel. This is equal 4*hydraulic_radius, with hydraulic_radius = channel_area / channel_perimeter.

        Args:
            x (float, optional): Axial position along the engine. This parameter may have no effect on the output. Defaults to None.
            y (float, optional): The radius of the engine (m) (NOT the radius of the cooling channel).  Only required for 'vertical' channels. 

        Returns:
            float: Effective diameter (m)
        """
        if self.configuration == 'spiral':
            return self.effective_diameter

        elif self.configuration == 'vertical':
            perimeter = 2*np.pi*y + 2*np.pi*(y + self.channel_height)
            return 4*self.A(x, y)/perimeter

        else:
            raise ValueError(f"The cooling jacket configuration {self.configuration} is not recognised. Try 'spiral' or 'vertical'. ")

    def coolant_velocity(self, rho_coolant, x = None, y = None):
        """Get coolant velocity using mdot = rho*V*A.

        Args:
            rho_coolant (float): Coolant density (kg/m^3)
            x (float, optional): x position - does not currently affect anything.
            y (float, optional): Is The radius of the engine (m) (NOT the radius of the cooling channel). Only required for 'vertical' channels. 

        Returns:
            float: Coolant velocity (m/s)
        """
        return self.mdot_coolant/(rho_coolant * self.A(x, y))

class Ablative:
    """Container for refractory or ablative properties. 

    Args:
        ablative_material (Material): Ablative material.
        regression_rate (float): (Not currently used) (m/s)
        xs (list, optional): x positions that the ablative is present between, [xmin, xmax]. Defaults to [-1000, 1000].
        wall_material (Material): Wall material on the outside of the ablative (will override the cooling jacket wall material).
        ablative_thickness (float or list): Thickness of ablative. If a list is given, it must correspond to thickness at regular x intervals, which will be stretched out over the inverval of 'xs'. Defaults to None (in which case the ablative extends from the engine contour to combustion chamber radius).
    """
    def __init__(self, ablative_material, wall_material, xs = [-1000, 1000], ablative_thickness = None, regression_rate = 0.0):
        self.ablative_material = ablative_material
        self.wall_material = wall_material
        self.regression_rate = regression_rate
        self.xs = xs
        
        if type(ablative_thickness) is float or type(ablative_thickness) is int:
            self.ablative_thickness = [ablative_thickness]  #Convert into a list so the interpolation works
        else:
            self.ablative_thickness = ablative_thickness