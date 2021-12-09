"""
Classes and functions related to thermal circuit calculations

References:
    - [1] - The Thrust Optimised Parabolic nozzle, AspireSpace, http://www.aspirespace.org.uk/downloads/Thrust%20optimised%20parabolic%20nozzle.pdf   \n
    - [2] - Rocket Propulsion Elements, 7th Edition  \n
    - [3] - Design and analysis of contour bell nozzle and comparison with dual bell nozzle https://core.ac.uk/download/pdf/154060575.pdf 
    - [4] - Modelling ablative and regenerative cooling systems for an ethylene/ethane/nitrous oxide liquid fuel rocket engine, Elizabeth C. Browne, https://mountainscholar.org/bitstream/handle/10217/212046/Browne_colostate_0053N_16196.pdf?sequence=1&isAllowed=y  \n
    - [5] - Thermofluids databook, CUED, http://www-mdp.eng.cam.ac.uk/web/library/enginfo/cueddatabooks/thermofluids.pdf    \n
    - [6] - A Simple Equation for Rapid Estimation of Rocket Nozzle Convective Heat Transfer Coefficients, Dr. R. Bartz, https://arc.aiaa.org/doi/pdf/10.2514/8.12572
    - [7] - Regenerative cooling of liquid rocket engine thrust chambers, ASI, https://www.researchgate.net/profile/Marco-Pizzarelli/publication/321314974_Regenerative_cooling_of_liquid_rocket_engine_thrust_chambers/links/5e5ecd824585152ce804e244/Regenerative-cooling-of-liquid-rocket-engine-thrust-chambers.pdf  \n
"""

import numpy as np

def h_gas_bartz(D, cp_inf, mu_inf, Pr_inf, rho_inf, v_inf, rho_am, mu_am, mu0):
    """
    Bartz equation, using Equation (8-23) from page 312 of RPE 7th edition (Reference [2]). 'am' refers to the gas being at the 'arithmetic mean' of the wall and freestream temperatures.

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

    Returns:
        float: Convective heat transfer coefficient
    """
    ReD = rho*V*D/mu
    NuD = (f/8) * (ReD - 1000) * Pr / (1 + 12.7*(f/8)**(1/2) *(Pr**(2/3) - 1))
    h = NuD * k / D
    return h

class ThermalCircuit:
    def __init__(self, T1, T2, R):
        """Class for solving thermal circuits. Will solve them upon initialising.

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
