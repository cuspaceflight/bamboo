"""
Classes and functions related to thermal circuit calculations.

References:
    - [1] - Rocket Propulsion Elements, 7th Edition 
    - [2] - Modelling ablative and regenerative cooling systems for an ethylene/ethane/nitrous oxide liquid fuel rocket engine, Elizabeth C. Browne, https://mountainscholar.org/bitstream/handle/10217/212046/Browne_colostate_0053N_16196.pdf?sequence=1&isAllowed=y  \n
    - [3] - A Simple Equation for Rapid Estimation of Rocket Nozzle Convective Heat Transfer Coefficients, Dr. R. Bartz, https://arc.aiaa.org/doi/pdf/10.2514/8.12572
    - [4] - https://en.wikipedia.org/wiki/Nucleate_boiling
    - [5] - https://en.wikipedia.org/wiki/Fin_(extended_surface)
    - [6] - Welty, Fundamentals of Momentum, Heat and Mass Transfer, Fifth Edition
"""

import numpy as np
import warnings

GRAVITY = 9.80665

# Free convection
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
        float: Convective heat transfer coefficient (W/m2/K), h, for the exhaust gas side (where q = h(T - T_inf)).
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
        float: Convective heat transfer coefficient (W/m2/K), h, for the exhaust gas side (where q = h(T - T_inf)).
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
        float: Convective heat transfer coefficient (W/m2/K)
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
        float: Convective heat transfer coefficient (W/m2/K)
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
        float: Convective heat transfer coefficient (W/m2/K)
    """
    ReD = rho*V*D/mu

    if ReD <= 1000:
        raise ValueError("Gnielinski correlation will give negative convective heat transfer coefficients for ReD < 1000")

    NuD = (f_darcy/8) * (ReD - 1000) * Pr / (1 + 12.7*(f_darcy/8)**(1/2) *(Pr**(2/3) - 1))
    h = NuD * k / D
    return h


# Fins
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


# Nucleate boiling
def dQ_dA_nucleate(mu_l, h_fg, rho_l, rho_v, sigma, cp_l, T_w, T_sat, C_sf, Pr_l):
    """Get the heat flux due to nucleate boiling. From Rohsenow's equation [4][6].

    Args:
        mu_l (float): Viscosity of the liquid phase (Pa s)
        h_fg (float): Enthalpy between vapour and liquid phases. h_fg = h_g - h_f. (J/kg/K)
        rho_l (float): Density of the liquid phase (kg/m3)
        rho_v (float): Density of the vapour phase (kg/m3)
        sigma (float): Surface tension of the liquid-vapour interface (N/m)
        cp_l (float): Isobaric specific heat capacity of the liquid (J/kg/K)
        T_w (float): Wall temperature (K)
        T_sat (float): Saturation temperature of the fluid (K)
        C_sf (float): Surface-fluid coefficient. Will be different for different material + fluid combinations. Some examples are available in [4] and [6].
        Pr_l (float): Prandtl number of the liquid phase

    Returns:
        float: Heat flux (W/m2)
    """
    return mu_l * h_fg * (GRAVITY * (rho_l - rho_v) / sigma)**0.5 * (cp_l * (T_w - T_sat) / (C_sf * h_fg * Pr_l**1.7))**3

def dQ_dA_nucleate_critical(h_fg, rho_v, sigma, rho_l):
    """Get the critical heat flux due to nucleate boiling, i.e. the maximum heat transfer rate that is possible. From Rohsenow's equation [4][6].

    Args:
        h_fg (float): Enthalpy between vapour and liquid phases. h_fg = h_g - h_f. (J/kg/K)
        rho_v (float): Density of the vapour phase (kg/m3)
        sigma (float): Surface tension of the liquid-vapour interface (N/m)
        rho_l (float): Density of the liquid phase (kg/m3)

    Returns:
        float: Heat flux (W/m2)
    """
    return 0.18 * h_fg * rho_v * ( (sigma * GRAVITY * (rho_l - rho_v)) / (rho_v**2) )**0.25

def h_coolant_stable_film(k_vf, rho_vf, rho_v, rho_l, h_fg, cp_l, dT, mu_vf, T_w, T_sat, sigma):
    """Convective heat transfer coefficient for the stable-film phase of boiling heat transfer [6]. The film temperature is defined as the mean of the wall and freestream temperature, i.e. 0.5 * (T_w + T_bulk)
       
    Args:
        k_vf (float): Thermal conductivity of the vapour, evaluated at the film temperature (W/m/K)
        rho_vf (float): Density of the vapour, evaluated at the film temperature (kg/m3)
        rho_v (float): Density of the vapour, evaluated at the bulk temperature? (kg/m3)
        rho_l (float): Density of the liquid, evaluated at the bulk temperature? (kg/m3)
        h_fg (float): Enthalpy between vapour and liquid phases. h_fg = h_g - h_f. (J/kg/K)
        cp_l (float): Isobaric specific heat capacity of the liquid (J/kg/K)
        dT (float): Temperature difference between the wall and bulk (T_w - T_freestream) (K)
        mu_vf (float): Viscosity of the vapour, evaluated at the film temperature
        T_w (float): Wall temperature (K)
        T_sat (float): Saturated vapour temperature (K)
        sigma (float): Surface tension of the liquid-vapour interface (N/m)

    Returns:
        float: Convective heat transfer coefficient (W/m2/K)
    """
    return 0.425 * ( k_vf**3 * rho_vf * (rho_l - rho_v) * GRAVITY * (h_fg + 0.4 * cp_l * dT) / (mu_vf * (T_w - T_sat) * (sigma / (GRAVITY * (rho_l - rho_v)) )**0.5 ) )**0.25


# Classes
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
