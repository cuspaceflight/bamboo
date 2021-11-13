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
    - [6] - A Simple Equation for Rapid Estimation of Rocket Nozzle Convective Heat Transfer Coefficients, Dr. R. Bartz, https://arc.aiaa.org/doi/pdf/10.2514/8.12572
    - [7] - Regenerative cooling of liquid rocket engine thrust chambers, ASI, https://www.researchgate.net/profile/Marco-Pizzarelli/publication/321314974_Regenerative_cooling_of_liquid_rocket_engine_thrust_chambers/links/5e5ecd824585152ce804e244/Regenerative-cooling-of-liquid-rocket-engine-thrust-chambers.pdf  \n
'''

import bamboo as bam
import numpy as np
import matplotlib.pyplot as plt
import scipy

SIGMA = 5.670374419e-8      #Stefan-Boltzmann constant (W/m^2/K^4)


def black_body(T):
    """Get the black body radiation emitted over a hemisphere, at a given temperature.

    Args:
        T (float): Temperature of the body (K)

    Returns:
        float: Radiative heat transfer rate, per unit emitting area on the body (W/m^2)
    """
    return SIGMA*T**4

def h_gas_rpe(D, M, T, rho, gamma, R, mu, k, Pr):
    """Get the convective heat transfer coefficient on the gas side. Uses Eqn (8-22) on page 312 or RPE 7th edition (Reference [2]). I believe this is just a form of the Dittius-Boelter equation.
    
    Note:
        Seems to give much lower wall temperatures than the Bartz equation, and is likely less accurate. h_gas_2 and h_gas_3 are likely more accurate.

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

def h_gas_bartz(D, cp_inf, mu_inf, Pr_inf, rho_inf, v_inf, rho_am, mu_am, mu0):
    """Bartz equation, 
    using Equation (8-23) from page 312 of RPE 7th edition (Reference [2]). 'am' refers to the gas being at the 'arithmetic mean' of the wall and freestream temperatures.

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

def h_gas_bartz_sigma(c_star, At, A, pc, Tc, M, Tw, mu0, cp0, gamma, Pr0):
    """Bartz heat transfer equation using the sigma correlation, from Reference [6].

    Args:
        c_star (float): C* efficiency ( = pc * At / mdot)
        At (float): Throat area (m^2)
        A (float): Flow area (m^2)
        pc (float): Chamber pressure (Pa)
        Tc (float): Chamber temperature (K)
        M (float): Freestream Mach number
        Tw (float): Wall temperature (K)
        mu0 (float): Absolute viscosity at stagnation conditions (Pa s)
        cp0 (float): Gas specific heat capacity at stagnation conditions (J/kg/K)
        gamma (float): Gas ratio of specific heats (cp/cv)
        Pr0 (float): Prandtl number at stagnation conditions

    Returns:
        float: Convective heat transfer coefficient, h, for the exhaust gas side (where q = h(T - T_inf)).
    """

    Dt = (At *4/np.pi)**0.5
    sigma = (0.5 * (Tw/Tc) * (1 + (gamma-1)/2 * M**2) + 0.5)**(-0.68) * (1 + (gamma-1)/2 * M**2)**(-0.12)

    return (0.026)/(Dt**0.2) * (mu0**0.2*cp0/Pr0**0.6) * (pc/c_star)**0.8 * (At/A)**0.9 * sigma

def h_coolant_rpe(A, D, mdot, mu, k, c_bar, rho):
    """Get the convective heat transfer coefficient for the coolant side.
    Uses the equation from page 317 of RPE 7th edition (Reference [2]).

    Args:
        A (float): Coolant flow area (m^2)
        D (float): Coolant channel hydraulic diameter (m)
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

def h_coolant_sieder_tate(rho, V, D, mu_bulk, mu_wall, Pr, k):
    """Sieder-Tate equation for convective heat transfer coefficient.

    Args:
        rho (float): Coolant bulk density (kg/m^3)
        V (float): Coolant bulk velocity (m/s)
        D (float): Hydraulic diameter of pipe (m)
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

def h_coolant_dittus_boelter(rho, V, D, mu, Pr, k):
    """Dittus-Boelter equation for convective heat transfer coefficient.

    Args:
        rho (float): Coolant bulk density (kg/m^3).
        V (float): Coolant bulk velocity (m/s)
        D (float): Hydraulic diameter of pipe (m)
        mu (float): Coolant bulk viscosity (Pa s)
        Pr (float): Coolant bulk Prandtl number
        k (float): Coolant thermal conductivity

    Returns:
        float: Convective heat transfer coefficient
    """
    Re = rho*V*D/mu
    Nu = 0.023*Re**(4/5)*Pr**0.4

    return Nu*k/D

def h_coolant_gnielinski(rho, V, D, mu, Pr, k, f):
    """Convective heat transfer coefficient for the coolant side, using Gnielinski's correlation. Page 41 of Reference [4].

    Args:
        rho (float): Coolant density (kg/m3)
        V (float): Coolant velocity (m/s)
        D (float): Hydraulic diameter (m)
        mu (float): Coolant viscosity (Pa s)
        Pr (float): Prandtl number
        k (float): Coolant thermal conductivity
        f (float): Coolant friction factor
    """
    ReD = rho*V*D/mu
    NuD = (f/8) * (ReD - 1000) * Pr / (1 + 12.7*(f/8)**(1/2) *(Pr**(2/3) - 1))
    h = NuD * k / D
    return h

class Material:
    """Class used to specify a material and its properties. For temperatures, only 'k' must be defined. For stresses, you also need E, alpha, and poisson.

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
        c (float, optional): Specific heat capacity (J/kg/K). Only available if assigned.
        rho (float, optional): Density (kg/m^3). Only available if assigned.
        Tsigma_coeffs (list, optional): List of coefficients, in ascending power order,
                                        for the polynomial for the temp / relative strength
                                        relationship.
        Tsigma_range (list, optional): Start, end temp (Kelvin) for Tsigma polynomial.
                                       At T = T_start, yield strength is sigma_y.
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

        if "Tsigma_coeffs" in kwargs and "Tsigma_range" in kwargs:
            self.polyCoeffs = kwargs["Tsigma_coeffs"]
            self.polyOrder = len(self.polyCoeffs)
            self.polyTmin = kwargs["Tsigma_range"][0]
            self.polyTmax = kwargs["Tsigma_range"][1]
            self.polyFlag = True # A valid relationship exists
            self.warned = False # Used to check if the user has been warned of inaccurate results, if they also set accuracy overrides
        else:
            print("Warning: Missing or invalid temperature-strength relationship. Stress results invalid for at least one material.")

        self.perf_therm = (1 - self.poisson) * self.k / (self.alpha * self.E)   #Performance coefficient for thermal stress, higher is better

    def __repr__(self):
        return f"""bamboo.cooling.Material Object
        Young's modulus = {self.E/1e9} GPa 
        0.2% Yield Stress = {self.sigma_y/1e6} MPa 
        Poisson's ratio = {self.poisson}
        alpha = {self.alpha} strain/K
        Thermal conductivity = {self.k} W/m/K
        (may also have a specific heat capacity (self.c) and density (self.rho))"""

    def relStrength(self, T, ignoreLowTemp = False, ignoreHighTemp = False):
        """Uses polynomial coefficients to determine the fraction of yield stress
           at a given temperature.

        Args:
            T (float): Temperature at which to find the relative strength
            ignoreLowTemp (bool, optional): If true and a temperature below the minimum
            specified by Tsigma_config is passed in, the relative strength at the minimum
            temperature is returned. Else an exception is raised. Defaults to False.
            ignoreHighTemp (bool, optional): If true and a temperature above the maximum
            specified by Tsigma_config is passed in, the relative strength at the maximum
            temperature is returned. Else an exception is raised. NOT RECOMMENDED. Defaults to False.

        Returns:
            float: A fraction of sigma_y
        """
        if self.polyFlag is False:
            raise ValueError("No valid material yield strength relationship for calculation.")

        if T > self.polyTmax:
            if ignoreHighTemp is False:
                raise ValueError(f"Material temperature out of bounds and override not set: {T} > {self.polyTmax}")
            if self.warned is False:
                print("Accuracy warning: Material temperature out of bounds at least once, continuing with override.")
                self.warned = True 
                # Set this flag true so the output isn't spammed - likely that more than one value will be out of range

        if T < self.polyTmin:
            if ignoreLowTemp is False:
                raise ValueError(f"Material temperature out of bounds and override not set: {T} < {self.polyTmin}")
            if self.warned is False:
                print("Accuracy warning: Material temperature out of bounds at least once, continuing with override.")
                self.warned = True 
                # Set this flag true so the output isn't spammed - likely that more than one value will be out of range

        return np.sum([self.polyCoeffs[index] * T**index for index in range(self.polyOrder)])

class TransportProperties:
    def __init__(self, type, Pr, mu, k, cp = None, rho = None, **kwargs):
        """Container for specifying your transport properties. If using type = "constants", submit a float for each argument. 
        If using type = "functions", the function arguments (except for the T_from_enthalpy function) must be in the form func(T, p) where T is the temperature in K and p is pressure in Pa.
        
        Note: 
            If you specify "T_from_h", this will be used to calculate temperature rises, instead of using cp. This can allow you to take into account latent heats of vapourisation, if desired.

        Note: 
            The reference point for enthalpies is arbritrary, but 'enthalpy_from_T' and 'T_from_enthalpy' must use the same reference point (so the two have the same enthalpy for the same temperature).

        Args:
            type (str): "constants" or "functions"
            Pr (float or function): Prandtl number.
            mu (float or function): Absolute viscosity (Pa s).
            k (float or function): Thermal conductivity (W/m/K).
            cp (float or function, optional): Isobaric specific heat capacity (J/kg/K) - only required for coolants.
            rho (float or function, optional): Density (kg/m^3) - only required for coolants.

        Keyword Args:
            T_from_enthalpy (function): Function to return temperature (K) if given specific enthalpy (J/kg) and pressure (Pa) in that order. i.e. func(enthalpy, p).
            enthalpy_from_T (function): Get specific enthalpy (J/kg) for a given temperature (K) and pressure (Pa). Takes arguments in the form func(T, p)
        """
        if type != "constants" and type != "functions":
            raise ValueError("Argument for transport properties 'type' can only be 'constants' or 'functions'.")

        self.type = type
        self.given_Pr = Pr
        self.given_mu = mu
        self.given_k = k
        self.given_rho = rho
        self.given_cp = cp

        if "T_from_enthalpy" in kwargs and "enthalpy_from_T" in kwargs:
            self.given_T_from_enthalpy = kwargs["T_from_enthalpy"]
            self.given_enthalpy_from_T = kwargs["enthalpy_from_T"]

        elif "T_from_enthalpy" in kwargs or "enthalpy_from_T" in kwargs: 
            raise ValueError("Must specify both T_from_enthalpy and enthalpy_from_T, or neither. Can't just specify one of them")

    
    def Pr(self, T, p):
        """Prandtl number.

        Args:
            T (float): Temperature (K)
            p (float): Pressure (Pa)

        Returns:
            float: Prandtl number
        """
        if self.type == "constants":
            return self.given_Pr

        elif self.type == "functions":
            return self.given_Pr(T, p)

    def mu(self, T, p):
        """Absolute viscosity (Pa s)

        Args:
            T (float): Temperature (K)
            p (float): Pressure (Pa)

        Returns:
            float: Absolute viscosity (Pa s)
        """
        if self.type == "constants":
            return self.given_mu

        elif self.type == "functions":
            return self.given_mu(T, p)

    def k(self, T, p):
        """Thermal conductivity (W/m/K)

        Args:
            T (float): Temperature (K)
            p (float): Pressure (Pa)

        Returns:
            float: Thermal conductivity (W/m/K)
        """
        if self.type == "constants":
            return self.given_k

        elif self.type == "functions":
            return self.given_k(T, p)

    def rho(self, T, p):
        """Density (kg/m^3)

        Args:
            T (float): Temperature (K)
            p (float): Pressure (Pa)

        Returns:
            float: Density (kg/m^3)
        """
        if self.type == "constants":
            return self.given_rho

        elif self.type == "functions":
            return self.given_rho(T, p)

    def cp(self, T, p):
        """Isobaric specific heat capacity (J/kg/K)

        Args:
            T (float): Temperature (K)
            p (float): Pressure (Pa)

        Returns:
            float: Isobaric specific heat capacity (J/kg/K)
        """
        if self.type == "constants":
            return self.given_cp

        elif self.type == "functions":
            return self.given_cp(T, p)

    def T_from_enthalpy(self, enthalpy, p):
        """Get temperature for a given specific enthalpy and pressure.

        Args:
            enthalpy (float): Specific enthalpy (J/kg)
            p (float): Pressure (Pa)

        Returns:
            float: Temperature (K)
        """
        return self.given_T_from_enthalpy(enthalpy, p)

    def enthalpy_from_T(self, T, p):
        """Get specific enthalpy from a given temperature and pressure.

        Args:
            T (float): Temperature (K)
            p (float): Pressure (Pa)

        Returns:
            float: Specific enthalpy (J/kg)
        """
        return self.given_enthalpy_from_T(T, p)

class ThermalCircuit:
    def __init__(self, T1, T2, R):
        """Class for solving thermal circuits.

        Args:
            T1 (float): Temperature at start
            T2 (float): Temperature at end
            R (list): List of resistances between T1 and T2

        Attributes:
            Qdot (float): Heat transfer rate (positive in the direction of T1 --> T2)
            T (list): List of temperatures in between each resistance, including T1 and T2 at either end. i.e. [T1, ..., T2].
        """
        self.R = R
        self.T1 = T1
        self.T2 = T2

        self.Qdot = (T1 - T2)/sum(R)
        self.T = np.zeros(len(R) + 1)
        self.T[0] = T1

        for i in range(1, len(R)):
            self.T[i] = self.T[i-1] - self.Qdot*R[i-1]

class CoolingJacket:
    """Container for cooling jacket information - e.g. for regenerative cooling. All channels are assumed to have rectangular cross sections.

    Args:
        inner_wall_material (Material): Wall material on the inner side of the cooling jacket.
        inlet_T (float): Inlet coolant temperature (K)
        inlet_p0 (float): Inlet coolant stagnation pressure (Pa)
        coolant_transport (TransportProperties): Container for the coolant transport properties.
        mdot_coolant (float): Coolant mass flow rate (kg/s)
        xs (list): x positions that the cooling jacket starts and ends at, [x_min, x_max].
        configuration (str, optional): Options include 'spiral' and 'vertical'. Defaults to "vertical".
        has_ablative (bool, optional): Whether or not the engine has an ablative.
    
    Keyword Args:
        blockage_ratio (float): This is the proportion (by area) of the channel cross section occupied by ribs.
        number_of_ribs (int): Only relevant if 'blockage_ratio' !=0. This is the number of ribs present in the cooling channel. For spiral channels this is the number of ribs 'per pitch' - it is numerically equal to the number of parallel spiral channels.
        channel_height (float or list): This is the height of the channels, in the radial direction (m). Can be a constant (float) or list. If list, it must be an channel heights at regularly spaced x positions, which will be stretched to fit the xs input.
        channel_width (float): Only relevant if configuration = 'spiral'. This is the total width (i.e. pitch) of the cooling channels (m). Can be a constant (float) or list. If list, it must be an channel heights at regularly spaced x positions, which will be stretched to fit the xs input.
        outer_wall_material (Material): Wall material for the outer liner.
    """
    def __init__(self, inner_wall_material, inlet_T, inlet_p0, coolant_transport, mdot_coolant, xs, configuration = "spiral", **kwargs):
        self.inner_wall_material = inner_wall_material
        self.coolant_transport = coolant_transport       
        self.mdot_coolant = mdot_coolant
        self.xs = xs
        self.inlet_T = inlet_T
        self.inlet_p0 = inlet_p0
        self.configuration = configuration
        self.channel_height_input = kwargs["channel_height"]

        if "outer_wall_material" in kwargs:
            self.outer_wall_material = kwargs["outer_wall_material"]
        
        if self.configuration == 'spiral':
            #Blockage ratio
            if "blockage_ratio" in kwargs:
                assert kwargs["blockage_ratio"] < 1, "Blockage ratio must be less than 1"
                self.blockage_ratio = kwargs["blockage_ratio"]
                
                if "number_of_ribs" in kwargs:
                    if type(kwargs["number_of_ribs"]) is not int:
                        raise ValueError("Keyword argument 'number_of_ribs' must be an integer")
                    elif kwargs["number_of_ribs"] < 1:
                        raise ValueError("Keyword argument 'number_of_ribs' must be at least 1 for spiral channels (it is numerically equal to the number of parallel spiralling channels).")
                    else:
                        self.number_of_ribs =  kwargs["number_of_ribs"]
                else:
                    raise ValueError("Must also specify 'number_of_ribs' if you want to specify 'blockage_ratio'")

            else:
                self.blockage_ratio = 0.0
                self.number_of_ribs = 1

            self.channel_width_input = kwargs["channel_width"]

        
        elif self.configuration == 'vertical':
            #Blockage ratio
            if "blockage_ratio" in kwargs:
                assert kwargs["blockage_ratio"] < 1, "Blockage ratio must be less than 1"
                self.blockage_ratio = kwargs["blockage_ratio"]
                
                if "number_of_ribs" in kwargs:
                    if type(kwargs["number_of_ribs"]) is not int:
                        raise ValueError("Keyword argument 'number_of_ribs' must be an integer")
                    else:
                        self.number_of_ribs =  kwargs["number_of_ribs"]
                else:
                    raise ValueError("Must also specify 'number_of_ribs' if you want to specify 'blockage_ratio'")

            else:
                self.blockage_ratio = 0.0
                self.number_of_ribs = 0

    def channel_height(self, x):
        """Get channel height at a given x postiion. Will interpolate if self.channel_height_input is a list.

        Args:
            x (float): x position in engine (m).

        Returns:
            float: Cooling channel height (m).
        """
        if type(self.channel_height_input) is float or type(self.channel_height_input) is int:
            return self.channel_height_input
        else:
            x_array = np.linspace(self.xs[0], self.xs[1], len(self.channel_height_input))

            return np.interp(x, x_array, self.channel_height_input)

    def channel_width(self, x):
        """Get channel width at a given x postiion. Will interpolate if self.channel_height_input is a list. Only works for spiral channels.

        Args:
            x (float): x position in engine (m).

        Returns:
            float: Cooling channel width (m).
        """
        if type(self.channel_width_input) is float or type(self.channel_width_input) is int:
            return self.channel_width_input
        else:
            x_array = np.linspace(self.xs[0], self.xs[1], len(self.channel_width_input))

            return np.interp(x, x_array, self.channel_width_input)

    def A(self, x, y):
        """Get coolant channel cross flow cross sectional area.

        Args:
            x (float, optional): x position - does not currently affect anything.
            y (float, optional): y distance from engine centreline to the inner wall of the cooling channel (m).

        Returns:
            float: Cooling channel flow area (m^2)
        """

        if self.configuration == 'spiral':
            return self.channel_width(x)*self.channel_height(x)*(1 - self.blockage_ratio)

        elif self.configuration == 'vertical':
            return np.pi*((y + self.channel_height(x))**2 - y**2) * (1 - self.blockage_ratio)
    
        else:
            raise ValueError(f"The cooling jacket configuration {self.configuration} is not recognised. Try 'spiral' or 'vertical'. ")

    def D(self, x, y):
        """Hydraulic diameter of the cooling channel. This is equal 4*channel_area / wetted_channel_perimeter.

        Args:
            x (float, optional): Axial position along the engine. 
            y (float, optional): y distance from engine centreline to the inner wall of the cooling channel (m).

        Note:
            Not entirely sure if I calculated the perimeter correctly when including blockage ratio.

        Returns:
            float: Hydraulic diameter (m)
        """
        if self.configuration == 'spiral':
            perimeter = 2*self.channel_width(x) + 2*self.channel_height(x) + 2*self.channel_height(x)*self.number_of_ribs
            return 4 * self.A(x, y)/perimeter

        elif self.configuration == 'vertical':
            if self.blockage_ratio == 0.0:
                perimeter = 2*np.pi*y + 2*np.pi*(y + self.channel_height(x)) 
                return 4*self.A(x, y)/perimeter

            else:
                #Not entirely sure if I calculated the perimeter correctly with blockage ratio
                perimeter = (2*np.pi*y + 2*np.pi*(y + self.channel_height(x)))*(1 - self.blockage_ratio) + 2*self.number_of_ribs*self.channel_height(x)
                return 4*self.A(x, y)/perimeter

        else:
            raise ValueError(f"The cooling jacket configuration {self.configuration} is not recognised. Try 'spiral' or 'vertical'. ")

    def coolant_velocity(self, rho_coolant, x, y):
        """Get coolant velocity using mdot = rho*V*A.

        Args:
            rho_coolant (float): Coolant density (kg/m^3)
            x (float, optional): x position - does not currently affect anything.
            y (float, optional): y distance from engine centreline to the inner wall of the cooling channel (m).

        Returns:
            float: Coolant velocity (m/s)
        """

        return self.mdot_coolant/(rho_coolant * self.A(x, y))

class Ablative:
    """Container for refractory or ablative properties. 

    Args:
        ablative_material (Material): Ablative material.
        regression_rate (float): (Not currently used) (m/s)
        xs (list): x positions that the ablative is present between, [xmin, xmax]. 
        wall_material (Material): Wall material on the outside of the ablative (will override the cooling jacket wall material).
        ablative_thickness (float or list): Thickness of ablative. If a list is given, it must correspond to thickness at regular x intervals, which will be stretched out over the inverval of 'xs'. Defaults to None (in which case the ablative extends from the engine contour to combustion chamber radius).
    """
    def __init__(self, ablative_material, wall_material, xs, ablative_thickness = None, regression_rate = 0.0):
        self.ablative_material = ablative_material
        self.wall_material = wall_material
        self.regression_rate = regression_rate
        self.xs = xs
        
        if type(ablative_thickness) is float or type(ablative_thickness) is int:
            self.ablative_thickness = [ablative_thickness]  #Convert into a list so the interpolation works
        else:
            self.ablative_thickness = ablative_thickness
