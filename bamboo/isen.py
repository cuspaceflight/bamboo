"""
Isentropic compressible flow relations.
 - [1] - Heister et al., Rocket Propulsion (https://doi.org/10.1017/9781108381376)
"""

import scipy.optimize

def m_bar(M, gamma):    
    """Non-dimensional mass flow rate, defined as m_bar = mdot * sqrt(cp*T0)/(A*p0). A is the local cross sectional area that the flow is moving through.

    Args:
        M (float): Mach number
        gamma (float): Ratio of specific heats cp/cv

    Returns:
        float: Non-dimensional mass flow rate
    """
    return gamma/(gamma-1)**0.5 * M * (1+ M**2 * (gamma-1)/2)**(-0.5*(gamma+1)/(gamma-1))

def A_At(M, gamma):
    """Ratio of local flow area to the throat area, for a given Mach number [1].

    Args:
        M (float): Mach number
        gamma (float): Ratio of specific heats cp/cv
    """
    return 1/M * ( (2 + (gamma-1) * M**2) / (gamma + 1) )**( (gamma+1) / (2 * (gamma-1) ) )

def p0(p, M, gamma):
    """Get stagnation pressure from static pressure and Mach number [1]

    Args:
        p (float): Pressure (Pa)
        M (float): Mach number
        gamma (float): Ratio of specific heats cp/cv

    Returns:
        float: Stagnation pressure (Pa)
    """
    return p*(1 + M**2 * (gamma-1)/2)**(gamma/(gamma-1))

def T0(T, M, gamma):
    """Get the stangation temperature from the static temperature and Mach number [1]

    Args:
        T (float): Temperature (K)
        M (float): Mach number
        gamma (float): Ratio of specific heats cp/cv

    Returns:
        float: Stagnation temperature (K)
    """
    return T*(1+ M**2 * (gamma-1)/2)


def Tr(T, M, gamma, r):
    """Get the recovery temperature, which is the temperature you should use for convective heat transfer (i.e. q = h(Tr - Tw)). [1]

    Args:
        T (float): Freestream static tempreature (K)
        M (float): Mach number
        gamma (float): Ratio of specific heats cp/cv
        r (float): 'Recovery factor', usually equal to Pr^(1/3) for 'turbulent free boundary layers' [1]

    Returns:
        float: Recovery temperature
    """
    return T * (1 + (gamma - 1)/2 * r * M**2)

def M_from_p(p, p0, gamma):
    """Mach number from static pressure and stagnation pressure.

    Args:
        p (float): Static pressure (Pa)
        p0 (float): Stagnation pressure (Pa)
        gamma (float): Ratio of specific heats cp/cv

    Returns:
        float: Mach number
    """
    return ( (2/(gamma-1)) * ( (p/p0)**((gamma-1)/(-gamma)) - 1 ) )**0.5

def M_from_A_subsonic(A, A_t, gamma):
    """Get the Mach number from the local flow area, assuming subsonic flow.

    Args:
        A (float): Local area (m2)
        A_t (float): Throat area (m2)
        gamma (float): Ratio of specific heats cp/cv
    """

    def func_to_solve(Mach):
        return  A/A_t - A_At(M = Mach, gamma = gamma)
        
    return scipy.optimize.root_scalar(func_to_solve, bracket = [1e-10, 1], x0 = 0.5).root

def M_from_A_supersonic(A, A_t, gamma):
    """Get the Mach number from the local flow area, assuming supersonic flow.

    Args:
        A (float): Local area (m2)
        A_t (float): Throat area (m2)
        gamma (float): Ratio of specific heats cp/cv
    """
    def func_to_solve(Mach):
        return  A/A_t - A_At(M = Mach, gamma = gamma)

    return scipy.optimize.root_scalar(func_to_solve, bracket = [1, 500], x0 = 1).root

def T(T0, M, gamma):
    """Get local temperature from the Mach number and stagnation temperature.

    Args:
        T0 (float): Stagnation temperature (K)
        M (float): Local Mach number
        gamma (float): Ratio of specific heats cp/cv

    Returns:
        float: Temperature (K)
    """
    return T0*(1 + (gamma-1)/2 * M**2)**(-1)

def p(p0, M, gamma):
    """Get local pressure from the Mach number and stagnation pressure.

    Args:
        p0 (float): Stagnation pressure (Pa)
        M (float): Local Mach number
        gamma (float): Ratio of specific heats cp/cv

    Returns:
        float: Pressure (Pa)
    """
    return p0*(1 + (gamma-1)/2 * M**2)**(-gamma/(gamma-1))


def get_choked_mdot(perfect_gas, chamber_conditions, At):
    """Get the mass flow rate through a choked nozzle.

    Args:
        perfect_gas (PerfectGas): Exhaust gas leaving the combustion chamber.
        chamber_conditions (CombustionChamber): Combustion chamber.
        At (float): Throat area (m2)

    """

    return m_bar(M = 1, gamma = perfect_gas.gamma) * At * chamber_conditions.p0 / (perfect_gas.cp * chamber_conditions.T0)**0.5

def get_throat_area(perfect_gas, chamber_conditions, mdot):
    """Get the nozzle throat area, given the gas properties and combustion chamber conditions. Assumes perfect gas with isentropic flow.

    Args:
        perfect_gas (PerfectGas): Exhaust gas leaving the combustion chamber.
        chamber_conditions (CombustionChamber): Combustion chamber.
        mdot (float): Mass flow rate of exhaust gas (kg/s)

    Returns:
        float: Throat area (m^2)
    """
    return (mdot * (perfect_gas.cp*chamber_conditions.T0)**0.5 ) / (m_bar(M = 1, gamma = perfect_gas.gamma) * chamber_conditions.p0) 

def get_exit_area(perfect_gas, chamber_conditions, p_e, mdot):
    """Get the nozzle exit area, given the gas properties and combustion chamber conditions. Assumes perfect gas with isentropic flow.

    Args:
        perfect_gas (PerfectGas): Gas object.
        chamber_conditions (CombustionChamber): CombustionChamber object
        p_e (float): Exit pressure (Pa)

    Returns:
        float: Optimum nozzle exit area (Pa)
    """

    Me = M_from_p(p_e, chamber_conditions.p0, perfect_gas.gamma)
    return (mdot * (perfect_gas.cp*chamber_conditions.T0)**0.5 ) / (m_bar(Me, perfect_gas.gamma) * chamber_conditions.p0)
