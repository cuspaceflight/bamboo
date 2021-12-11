"""
Classes and functions related to thermal circuit calculations.

References (*need to clean up, not all are used here):
    - [1] - Rocket Propulsion Elements, 7th Edition 
    - [2] - Modelling ablative and regenerative cooling systems for an ethylene/ethane/nitrous oxide liquid fuel rocket engine, Elizabeth C. Browne, https://mountainscholar.org/bitstream/handle/10217/212046/Browne_colostate_0053N_16196.pdf?sequence=1&isAllowed=y  \n
    - [3] - A Simple Equation for Rapid Estimation of Rocket Nozzle Convective Heat Transfer Coefficients, Dr. R. Bartz, https://arc.aiaa.org/doi/pdf/10.2514/8.12572
    - [4] - https://en.wikipedia.org/wiki/Nucleate_boiling
    - [5] - https://en.wikipedia.org/wiki/Fin_(extended_surface)
"""

import numpy as np
import warnings


def h_gas_bartz(D, cp_inf, mu_inf, Pr_inf, rho_inf, v_inf, rho_am, mu_am, mu0):
    """
    Bartz equation, using Equation (8-23) from page 312 of RPE 7th edition (Reference [1]). 'am' refers to the gas being at the 'arithmetic mean' of the wall and freestream temperatures.

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

def h_gas_bartz_sigma(c_star, At, A, p_chamber, T_chamber, M, Tw, mu0, cp0, gamma, Pr0):
    """Bartz heat transfer equation using the sigma correlation, from Reference [3].

    Args:
        c_star (float): C* efficiency ( = pc * At / mdot)
        At (float): Throat area (m^2)
        A (float): Flow area (m^2)
        p_chamber (float): Chamber pressure (Pa)
        T_chamber (float): Chamber temperature (K)
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
    sigma = (0.5 * (Tw/T_chamber) * (1 + (gamma-1)/2 * M**2) + 0.5)**(-0.68) * (1 + (gamma-1)/2 * M**2)**(-0.12)

    return (0.026)/(Dt**0.2) * (mu0**0.2*cp0/Pr0**0.6) * (p_chamber/c_star)**0.8 * (At/A)**0.9 * sigma

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

def h_coolant_gnielinski(rho, V, D, mu, Pr, k, f_darcy):
    """Convective heat transfer coefficient for the coolant side, using Gnielinski's correlation. Page 41 of Reference [2].

    Args:
        rho (float): Coolant density (kg/m3)
        V (float): Coolant velocity (m/s)
        D (float): Hydraulic diameter (m)
        mu (float): Coolant viscosity (Pa s)
        Pr (float): Prandtl number
        k (float): Coolant thermal conductivity
        f_darcy (float): Darcy friction factor for the coolant

    Returns:
        float: Convective heat transfer coefficient
    """
    ReD = rho*V*D/mu

    if ReD <= 1000:
        raise ValueError("Gnielinski correlation will give negative convective heat transfer coefficients for ReD < 1000")

    NuD = (f_darcy/8) * (ReD - 1000) * Pr / (1 + 12.7*(f_darcy/8)**(1/2) *(Pr**(2/3) - 1))
    h = NuD * k / D
    return h

def h_coolant_rohsenow():
    raise ValueError("Rohsenow correlation not yet implemented")
    # Nucleate boiling correlation from Reference [4]

def Q_fin_adiabatic(P, Ac, k, h, L, T_b, T_inf):
    """Get the heat transfer rate for a fin with an adiabatic tip (Reference [5])

    Args:
        P (float): Fin perimeter (m)
        Ac (float): Fin cross sectional area (m2)
        k (float): Fin thermal conductivity (W/m/K)
        h (float): Convective heat transfer coefficient at the fin surface (W/m2/K)
        L (float): Fin length (m)
        T_b (float): Fin base temperature (K)
        T_inf (float): Freestream temperature of the fluid the fin is submersed in (K)

    Returns:
        float: Heat transfer rate out of fin (W)
    """
    m = np.sqrt(h * P / (k * Ac))
    theta_b = T_b - T_inf

    return np.sqrt(h * P * k * Ac) * theta_b * np.tanh(m * L)

class ThermalCircuit:
    def __init__(self, T1, T2, R):
        """Class for solving thermal circuits. Will solve them upon initialising.

        Args:
            T1 (float): Temperature at start
            T2 (float): Temperature at end
            R (list): List of resistances between T1 and T2, in the order [R_touching_T1, ... , R_touching_T2]

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
        self.T[-1] = T2

        for i in range(1, len(R)):
            self.T[i] = self.T[i-1] - self.Qdot*R[i-1]
