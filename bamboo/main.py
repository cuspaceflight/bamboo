'''
Simple script for calculating nozzle geometry from the chamber conditions. All units SI unless otherwise stated. All angles in radians unless otherwise stated.

Assumptions:
    - 1D flow.
    - Isentropic flow.
    - Perfect gases.

Conventions:
    - Position (x) along the nozzle is defined by: x = 0 at the throat, x < 0 in the combustion chamber, x > 0 in the diverging section of the nozzle.

Known issues:
    - I'm not sure if the Engine.optimise_for_apogee() function is working correctly.
    - A hardcoded fix is in place for using area ratios outside the Rao angle data range (it tricks the code into making a 15 degree cone). A more robust fix would be better.

Room for improvement:
    - Rename the "Gas" object as "PerfectGas", to avoid confusion with thermo module objects or any real gas models we might use later.
    - Rao bell nozzle data is currently obtained rather crudely (by using an image-of-graph-to-data converter). Would be nicer to have more exact data values.
    - Cone nozzles are not currently implemented.

Subscripts:
    0 - Stagnation condition
    c - Chamber condition (should be the same as stagnation conditions)
    t - At the throat
    e - At the nozzle exit plane
    amb - Atmopsheric/ambient condition

References:
    [1] - The Thrust Optimised Parabolic nozzle, AspireSpace, http://www.aspirespace.org.uk/downloads/Thrust%20optimised%20parabolic%20nozzle.pdf  
    [2] - Liquid rocket engine nozzles, NASA, https://ntrs.nasa.gov/api/citations/19770009165/downloads/19770009165.pdf  
    [3] - Design and analysis of contour bell nozzle and comparison with dual bell nozzle https://core.ac.uk/download/pdf/154060575.pdf
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import ambiance

'''Constants'''
R_BAR = 8.3144621e3         #Universal gas constant (J/K/kmol)
g0 = 9.80665                #Standard gravitational acceleration (m/s^2)

'''Compressible flow functions'''
def m_bar(M, gamma):    
    """Non-dimensional mass flow rate, defined as mdot * sqrt(cp*T0)/(A*p0). A is the local cross sectional area that the flow is moving through.

    Args:
        M (float): Mach number
        gamma (float): Ratio of specific heats cp/cv

    Returns:
        float: Non-dimensional mass flow rate
    """
    '''gamma is the ratio of specific heats, M is the Mach number '''
    return gamma/(gamma-1)**0.5 * M * (1+ M**2 * (gamma-1)/2)**(-0.5*(gamma+1)/(gamma-1))

def p0(p, M, gamma):
    """Get stagnation pressure from static pressure and Mach number

    Args:
        p (float): Pressure (Pa)
        M (float): Mach number
        gamma (float): Ratio of specific heats cp/cv

    Returns:
        float: Stagnation pressure (Pa)
    """
    return p*(1 + M**2 * (gamma-1)/2)**(gamma/(gamma-1))

def T0(T, M, gamma):
    """Get the stangation temperature from the static temperature and Mach number

    Args:
        T (float): Temperature (K)
        M (float): Mach number
        gamma (float): Ratio of specific heats cp/cv

    Returns:
        float: Stagnation temperature (K)
    """
    return T*(1+ M**2 * (gamma-1)/2)

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


'''Rocket trajectory functions'''
def estimate_apogee(dry_mass, propellant_mass, engine, cross_sectional_area, drag_coefficient = 0.75, dt = 0.2, show_plot = False):
    global g0

    #Use the convention that everything is positive upwards
    i = 1
    t = dt
    alts = [0.0]
    vels = [0.0]

    initial_mass = dry_mass + propellant_mass
    burn_time = propellant_mass/engine.chamber_conditions.mdot
    reached_apogee = False

    #Rate of change of the state array (f = [alt, vel])  
    def fdot(fn, t):          
        #The ambiance atmosphere model only goes up to 81020 m
        if fn[0] < 81020:
            density_amb = ambiance.Atmosphere(fn[0]).density[0]       
            p_amb = ambiance.Atmosphere(fn[0]).pressure[0] 
        else:
            density_amb = 0
            p_amb = 0

        #Calculate acceleration
        if t < burn_time:
            #Engine still on
            mass = initial_mass - engine.chamber_conditions.mdot*t
            try:
                net_force = engine.thrust(p_amb) - 0.5*density_amb*fn[1]**2*drag_coefficient*cross_sectional_area - mass*g0
            except ValueError:
                raise ValueError(f"Flow seperation occured in the nozzle at an altitude of {fn[0]/1000} km")
            acc = net_force/mass

        else:
            #Engine burnt out
            net_force = -0.5*density_amb*fn[1]**2*drag_coefficient*cross_sectional_area - dry_mass*g0
            acc = net_force/dry_mass

        return np.array([fn[1], acc], dtype = 'float')

    while reached_apogee == False:
        #RK4 implementation
        fn = np.array([alts[i-1], vels[i-1]])     #Rocket's state array

        k1 = fdot(fn, t)
        k2 = fdot(fn + k1*dt/2, t + dt/2)
        k3 = fdot(fn + k2*dt/2, t + dt/2)
        k4 = fdot(fn + k3*dt, t + dt)

        fnplusone = fn + (1/6)*(k1 + 2*k2 + 2*k3 + k4)*dt   # + O(dt^5) = [alt, vel]

        alts.append(fnplusone[0])
        vels.append(fnplusone[1])

        if alts[i] < alts[i-1]:
            apogee = alts[i]
            reached_apogee = True

        i = i + 1
        t = t + dt

    if show_plot == True:
        fig, axs = plt.subplots()
        axs.plot(np.linspace(0, t, len(alts)), alts)
        axs.set_xlabel("Time (s)")
        axs.set_ylabel("Altitude (m)")
        axs.grid()
        plt.show()

    return apogee


'''Nozzle geometry functions'''
def rao_theta_n(area_ratio, length_fraction = 0.8):
    """Returns the contour angle at the inflection point of the bell nozzle, by interpolating data.   
    Data obtained by using http://www.graphreader.com/ on the graph in Reference [1].

    Args:
        area_ratio (float): Area ratio of the nozzle (A2/At)
        length_fraction (int, optional): Nozzle contraction percentage, as defined in Reference [1]. Defaults to 0.8.
    
    Returns:
        float: "theta_n", angle at the inflection point of the bell nozzle (rad)
    """
    
    #Choose the data to use
    if length_fraction == 0.8:
        data = {"area_ratio":[3.678,3.854,4.037,4.229,4.431,4.642,4.863,5.094,5.337,5.591,5.857,6.136,6.428,6.734,7.055,7.391,7.743,8.111,8.498,8.902,9.326,9.77,10.235,10.723,11.233,11.768,12.328,12.915,13.53,14.175,14.85,15.557,16.297,17.074,17.886,18.738,19.63,20.565,21.544,22.57,23.645,24.771,25.95,27.186,28.48,29.836,31.257,32.746,34.305,35.938,37.649,39.442,41.32,43.288,45.349,47.508,49.77,52.14,54.623],
                "theta_n":[21.067,21.319,21.601,21.908,22.215,22.482,22.734,22.986,23.238,23.489,23.736,23.984,24.232,24.48,24.728,24.965,25.176,25.387,25.598,25.809,26.02,26.231,26.441,26.617,26.792,26.968,27.143,27.319,27.494,27.67,27.845,27.996,28.134,28.272,28.409,28.547,28.684,28.822,28.965,29.119,29.272,29.426,29.58,29.733,29.887,30.04,30.169,30.298,30.426,30.554,30.683,30.811,30.94,31.085,31.239,31.393,31.546,31.7,31.853]}
    else:
        raise ValueError("The length percent given does not match any of the available data.")
    
    #Make sure we're not outside the bounds of our data
    if area_ratio < 3.7 or area_ratio > 47:
        print("WARNING: Area ratio is outside the range of the Rao inflection angle data, returning 15 deg instead.")
        return 15.0*np.pi/180
        #raise ValueError(f"The area ratio provided ({area_ratio}) is outside of the range of available data. Maximum available is {data['area_ratio'][-1]}, minimum is {data['area_ratio'][0]}.")
    
    else:
        #Linearly interpolate and return the result, after converting it to radians.
        return np.interp(area_ratio, data["area_ratio"], data["theta_n"]) * np.pi/180

def rao_theta_e(area_ratio, length_fraction = 0.8):
    """Returns the contour angle at the exit of the bell nozzle, by interpolating data.  
    Data obtained by using http://www.graphreader.com/ on the graph in Reference [1].

    Args:
        area_ratio (float): Area ratio of the nozzle (A2/At)
        length_fraction (int, optional): Nozzle contraction percentage, as defined in Reference [1]. Defaults to 0.8.
    
    Returns:
        float: "theta_e", angle at the exit of the bell nozzle (rad)
    """
    
    #Choose the data to use
    if length_fraction == 0.8:
       data = {"area_ratio":[3.678,3.854,4.037,4.229,4.431,4.642,4.863,5.094,5.337,5.591,5.857,6.136,6.428,6.734,7.055,7.391,7.743,8.111,8.498,8.902,9.326,9.77,10.235,10.723,11.233,11.768,12.328,12.915,13.53,14.175,14.85,15.557,16.297,17.074,17.886,18.738,19.63,20.565,21.544,22.57,23.645,24.771,25.95,27.186,28.48,29.836,31.257,32.746,34.305,35.938,37.649,39.442,41.32,43.288,45.349,47.508],
               "theta_e":[14.355,14.097,13.863,13.624,13.372,13.113,12.889,12.684,12.479,12.285,12.096,11.907,11.733,11.561,11.393,11.247,11.101,10.966,10.832,10.704,10.585,10.466,10.347,10.229,10.111,10.001,9.927,9.854,9.765,9.659,9.553,9.447,9.341,9.235,9.133,9.047,8.962,8.877,8.797,8.733,8.67,8.602,8.5,8.398,8.295,8.252,8.219,8.187,8.155,8.068,7.96,7.851,7.744,7.68,7.617,7.553]}
            
    else:
        raise ValueError("The length percent given does not match any of the available data.")
    
    #Check if we're outside the bounds of our data
    if area_ratio < 3.7 or area_ratio > 47:
        print("WARNING: Area ratio is outside the range of the Rao exit angle data, returning 14.999 deg instead.")
        return 14.999*np.pi/180
        #raise ValueError(f"The area ratio provided ({area_ratio}) is outside of the range of available data. Maximum available is {data['area_ratio'][-1]}, minimum is {data['area_ratio'][0]}.")
    
    else:
        #Linearly interpolate and return the result, after converting it to radians
        return np.interp(area_ratio, data["area_ratio"], data["theta_e"]) * np.pi/180

def get_throat_area(gas, chamber_conditions):
    """Get the nozzle throat area, given the gas properties and combustion chamber conditions. Assumes perfect gas with isentropic flow.

    Args:
        gas (Gas): Exhaust gas leaving the combustion chamber.
        chamber_conditions (CombustionChamber): Combustion chamber.

    Returns:
        float: Throat area (m^2)
    """
    return (chamber_conditions.mdot * (gas.cp*chamber_conditions.T0)**0.5 )/( m_bar(1, gas.gamma) * chamber_conditions.p0) 

def get_exit_area(gas, chamber_conditions, p_amb):
    """Get the nozzle exit area, given the gas properties and combustion chamber conditions. Assumes perfect gas with isentropic flow.

    Args:
        gas (Gas): Gas object.
        chamber_conditions (CombustionChamber): CombustionChamber object
        p_amb (float): Ambient pressure (Pa)

    Returns:
        float: Optimum nozzle exit area (Pa)
    """

    Me = M_from_p(p_amb, chamber_conditions.p0, gas.gamma)
    return (chamber_conditions.mdot * (gas.cp*chamber_conditions.T0)**0.5 )/(m_bar(Me, gas.gamma) * chamber_conditions.p0)

def show_conical_shape(A1, At, A2, div_half_angle = 15, conv_half_angle=45):
    """Legacy function. Plots the shape of a conical nozzle with the specified half angle.

    Args:
        A1 (Chamber area): Chamber area (m^2)
        At (Throat area): Throat area (m^2)
        A2 (float) : Exit plane area (m^2)
        div_half_angle (float, optional): Cone half angle for the diverging section (deg). Defaults to 15.
        conv_half_angle (float, optional): Cone half angle for the converging section (deg). Defaults to 45.
    """

    #Convert angles to radians
    div_half_angle = div_half_angle*np.pi/180
    conv_half_angle = conv_half_angle*np.pi/180
    
    #Convert areas to radii
    r1 = (A1/np.pi)**0.5
    rt = (At/np.pi)**0.5
    r2 = (A2/np.pi)**0.5

    x = [0, (r1-rt)/np.tan(conv_half_angle), (r1-rt)/np.tan(conv_half_angle) + (r2-rt)/np.tan(div_half_angle)]
    y_pos = [r1, rt, r2]
    y_neg = [-r1, -rt, -r2]

    plt.plot(x, y_pos, color='b')
    plt.plot(x, y_neg, color='b')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("r1={:.5g} m, rt={:.5g} m, r2={:.5g} m".format(r1,rt,r2))
    plt.show()


'''Classes'''
class PerfectGas:
    """Object to store exhaust gas properties. Assumes a perfect gas (ideal gas with constant cp, cv and gamma). Only two properties need to be specified.

    Keyword Args:
        gamma (float): Ratio of specific heats cp/cv.
        cp (float): Specific heat capacity at constant pressure (J/kg/K)
        molecular_weight (float): Molecular weight of the gas (kg/kmol)
    """
    def __init__(self, **kwargs):
        if len(kwargs) > 2:
            raise ValueError(f"Gas object is overdefined. You mustn't provide more than 2 inputs when creating the Gas object. You provided {len(kwargs)}.")

        elif "gamma" in kwargs and "molecular_weight" in kwargs: 
            self.gamma = kwargs["gamma"]
            self.molecular_weight = kwargs["molecular_weight"]
            self.R = R_BAR/self.molecular_weight  
            self.cp = (self.gamma*self.R)/(self.gamma-1)    

        elif "gamma" in kwargs and "cp" in kwargs: 
            self.gamma = kwargs["gamma"]
            self.cp = kwargs["cp"]
            self.R = self.cp*(self.gamma-1)/self.gamma
            self.molecular_weight = R_BAR/self.R

        else:
            raise ValueError(f"Not enough inputs provided to fully define the Gas. You must provide exactly 2, but you provided only {len(kwargs)}.")

    def __repr__(self):
        return f"<nozzle.perfect_gas object> with: \ngamma = {self.gamma} \ncp = {self.cp} \nmolecular_weight = {self.molecular_weight} \nR = {self.R}"

class ChamberConditions:
    """Object for storing combustion chamber thermodynamic conditions.

    Args:
        p0 (float): Gas stagnation pressure (Pa).
        T0 (float): Gas stagnation temperature (K).
        mdot (float): Propellant mass flow rate (kg/s)
    """
    def __init__(self, p0, T0, mdot):
        self.p0 = p0
        self.T0 = T0
        self.mdot = mdot

class Nozzle:
    """Object for calculating and storing nozzle geometry.

    Args:
        At (float): Throat area (m^2).
        Ae (float): Exit plane area (m^2)
        type (str, optional): Desired shape, can be "rao" or "conical". Conical is not currently implemented. Defaults to "rao".
        length_fraction (float): Length fraction if a Rao nozzle is used. Defaults to 0.8.
        
    """
    def __init__(self, At, Ae, type = "rao", length_fraction = 0.8):
        self.At = At
        self.Ae = Ae
        self.Rt = (At/np.pi)**0.5   #Throat radius (m)
        self.Re = (Ae/np.pi)**0.5   #Exit radius (m)
        self.type = type
        self.length_fraction = length_fraction

        if self.type == "rao":
            self.theta_n = rao_theta_n(self.Ae/self.At)     #Inflection angle (rad), as defined in [1]
            self.theta_e = rao_theta_e(self.Ae/self.At)     #Exit angle (rad), as defined in [1]

            #Page 5 of Reference [1]:
            self.Nx = 0.382*self.Rt*np.cos(self.theta_n - np.pi/2)                              #Inflection point x-value
            self.Ny = 0.382*self.Rt*np.sin(self.theta_n - np.pi/2) + 0.382*self.Rt + self.Rt    #Inflection point y-value
            self.Ex = 0.8*((self.Re/self.Rt) - 1)*self.Rt/np.tan(np.pi/12)                      #Exit x-value
            self.Ey = self.Re                                                                   #Exit y-value (same as Re)
            self.length = self.Ex                                                               #Nozzle length

            #Page 2 of Reference [3]. Set up the matrix problem to get the coefficients for x = ay^2 + by + c
            A = np.array([[2*self.Ny, 1, 0],
                          [2*self.Ey, 1, 0],
                          [self.Ny**2, self.Ny, 1]], dtype='float')
            
            b = np.array([1/np.tan(self.theta_n), 1/np.tan(self.theta_e), self.Nx], dtype='float')

            self.a, self.b, self.c = np.linalg.solve(A, b)


        else:
            raise ValueError(f"Nozzle type '{type}' is not currently implemented. Try 'conical' or 'rao'")

    def __repr__(self):
        if self.type == "rao":
            return f"Rao type nozzle (length fraction = {self.length_fraction}). \nLength = {self.length} m \nThroat area = {self.At} m^2 \nExit area = {self.Ae} m^2 \nArea ratio = {self.Ae/self.At} \nRao inflection angle = {self.theta_n*180/np.pi} deg \nRao exit angle = {self.theta_e*180/np.pi} deg"

    def y(self, x):
        """Returns the distance between the nozzle contour and the centreline, given the axial distance 'x' downstream from the throat. Based on Reference [1] page 5.

        Args:
            x (float): Distance along the centreline from the throat (m)

        Returns:
            float: Distance between the nozzle centreline and the contour (m)
        """
        if x < 0:
            raise ValueError(f"x must be greater than zero. You tried to input {x}.")

        elif self.type == "rao" and x <= self.length:
            #Circular throat section
            if x < self.Nx:
                theta = -np.arccos(x/(0.382*self.Rt)) #Take the negative, because we want an answer in the range [-90 to 0 deg], but numpy gives us the one in the range [0 to 180 deg]
                return 0.382*self.Rt*np.sin(theta) + 0.382*self.Rt + self.Rt
            
            #Parabolic section.
            else:
                return ((4*self.a*(x-self.c) + self.b**2)**0.5 - self.b)/(2*self.a)   #Rearranging the quadratic on page 2 of Reference [3] to solve for y

        else:
            raise ValueError(f"x is beyond the end of the nozzle, which is only {self.length} m long. You tried to input {x}.")

    def A(self, x):
        """Returns the nozzle area given the axial distance 'x' downstream from the throat. Based on Reference [1] page 5.

        Args:
            x (float): Distance along the centreline from the throat (m)

        Returns:
            float: Nozzle cross sectional area (m^2) at the given value of x.
        """
        return np.pi*self.y(x)**2   #pi*R^2

    def plot_nozzle(self, number_of_points = 1000):
        if self.type == "rao":
            x = np.linspace(0, self.Ex, number_of_points)
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
            plt.show()

    @staticmethod
    def from_engine_components(gas, chamber_conditions, p_amb, type = "rao", length_fraction = 0.8):
        """Generate nozzle based on given gas properties and combustion chamber conditions

        Args:
            gas (Gas): Gas object for the exhaust gases.
            chamber_conditions (CombustionChamber): CombustionChamber object.
            p_amb (float): Ambient pressure (Pa). The nozzle will be designed to have this pressure at exit.
            type (str, optional): Nozzle type. Can be "rao" or "conical". Conical is not currently implemented. Defaults to "rao".
            length_fraction (float, optional): Rao nozzle length fraction, as defined in Reference [1]. Defaults to 0.8.

        Returns:
            [Nozzle]: The nozzle object.
        """
        return Nozzle(At = get_throat_area(gas, chamber_conditions), 
                    Ae = get_exit_area(gas, chamber_conditions, p_amb), 
                    type = type, length_fraction = length_fraction)

class Engine:
    """Class for representing a liquid rocket engine.

    Args:
        gas (PerfectGas): Gas representing the exhaust gas for the engine.
        chamber_conditions (CombustionChamber): CombustionChamber for the engine.
        nozzle (Nozzle): Nozzle for the engine.
    """
    def __init__(self, perfect_gas, chamber_conditions, nozzle):
        self.perfect_gas = perfect_gas
        self.chamber_conditions = chamber_conditions
        self.nozzle = nozzle

    def M(self, x):
        """Numerically solve to get the Mach number at a given distance x from the throat, along the centreline. 

        Args:
            x (float): Distance along centreline from the throat (m)
        """
        #If we're at the throat M=1 by default:
        if x==0:
            return 1.00

        #If we're not at the throat:
        else:
            def func_to_solve(Mach):
                return self.chamber_conditions.mdot*(self.perfect_gas.cp*self.chamber_conditions.T0)**0.5 / (self.nozzle.A(x)*self.chamber_conditions.p0) - m_bar(Mach, self.perfect_gas.gamma)
            
            Mach = scipy.optimize.root_scalar(func_to_solve, bracket = [1,25], x0 = 1).root

            return Mach

    def T(self, x):
        """Get temperature at a position along the nozzle.

        Args:
            x (float): Distance from the throat, along the centreline (m)

        Returns:
            float: Temperature (K)
        """
        return T(self.chamber_conditions.T0, self.M(x), self.perfect_gas.gamma)

    def p(self, x):
        """Get pressure at a position along the nozzle.

        Args:
            x (float): Distance from the throat, along the centreline (m)

        Returns:
            float: Pressure (Pa)
        """
        return p(self.chamber_conditions.p0, self.M(x), self.perfect_gas.gamma)

    def check_seperation(self, p_amb):
        """Approximate check for nozzle seperation. Based off page 17 of Reference [2].  
        Seperation occurs when P_wall/P_amb = 0.583 * (P_amb/P_chamber)^(0.195)

        Args:
            p_amb (float): Ambient (atmospheric) pressure.
        
        Returns:
            bool or float: Returns the position x (m) from the throat at which seperation occurs, if it does occur. If not, it returns False.
        """
        
        #Get the value of P_wall/P_amb requried for seperation
        seperation_pressure_ratio = 0.583 * (p_amb/self.chamber_conditions.p0)**0.195

        #Seperation can't occur if there's a vacuum outside:
        if p_amb == 0:
            return False

        #Check for seperation by comparing to the lowest pressure in the nozzle (which will be at exit):
        elif self.p(self.nozzle.length)/p_amb < seperation_pressure_ratio:
            def func_to_solve(x):
                return self.p(x)/p_amb - seperation_pressure_ratio  #Should equal zero at the seperation point
            
            return scipy.optimize.root_scalar(func_to_solve, bracket = [0, self.nozzle.length], x0 = 0).root    #Find the seperation position. 
            
        else:
            return False

    def seperation_p_amb(self):
        """Approximate way of getting the ambient pressure at which nozzle wall seperation occurs. Based off page 17 of Reference [2].  
        Seperation occurs when P_wall/P_amb = 0.583 * (P_amb/P_chamber)^(0.195). 
        It will first occur when P_e = P_wall satisfies this equation, since P_e is the lowest pressure in the nozzle.

        Returns:
            float: Ambient pressure at which seperation first occurs (Pa)
        """
        pc = self.chamber_conditions.p0
        pe = self.p(self.nozzle.length)
        return ((pe*pc**0.195)/0.583)**(1/1.195)

    def seperation_Ae(self, p_amb):
        """Approximate way of getting the exit area at which nozzle wall seperation occurs. Based off page 17 of Reference [2].  
        Seperation occurs when P_wall/P_amb = 0.583 * (P_amb/P_chamber)^(0.195). 

        Returns:
            float: Exit area at which seperation occurs for the given p_amb (m^2)
        """
        seperation_wall_pressure = p_amb * 0.583 * (p_amb/self.chamber_conditions.p0)**0.195

        #Get the exit area that gives this wall pressure at the exit.
        return get_exit_area(self.perfect_gas, self.chamber_conditions, seperation_wall_pressure)

    def thrust(self, p_amb):
        """Returns the thrust of the engine for a given ambient pressure.

        Args:
            p_amb (float): Ambient pressure (Pa)

        Returns:
            float: Thrust (N)
        """
        if self.check_seperation(p_amb) ==  False:
            Me = self.M(self.nozzle.length)
            Te = self.T(self.nozzle.length)
            pe = self.p(self.nozzle.length)

            return self.chamber_conditions.mdot*Me*(self.perfect_gas.gamma*self.perfect_gas.R*Te)**0.5 + (pe - p_amb)*self.nozzle.Ae    #Generic equation for rocket thrust
        
        else:
            raise ValueError(f"Seperation occured in the nozzle, at a postion {self.check_seperation(p_amb)} m downstream of the throat.")

    def isp(self, p_amb):
        """Returns the specific impulse for a given ambient pressure.

        Args:
            p_amb (float): Ambient pressure (Pa)

        Returns:
            float: Specific impulse (s)
        """
        global g0
        return self.thrust(p_amb)/(g0*self.chamber_conditions.mdot)

    def optimise_for_apogee(self, dry_mass, propellant_mass, cross_sectional_area, drag_coefficient = 0.75, dt = 0.2, debug=True):
        """Runs a 1D trajectory simulation, and varies the nozzle area ratio in an attempt to maximise apogee. Replaces the engine's nozzle with the optimised nozzle upon completion.

        Args:
            dry_mass (float): Dry mass of the launch vehicle (kg)
            propellant_mass (float): Initial mass of propellant inthe vehicle (kg)
            cross_sectional_area (float): Cross section area of the vehicle (used for calculating aerodynamic drag) (m^2)
            drag_coefficient (float, optional): Launch vehicle drag coefficient, assumed constant. Defaults to 0.75.
            dt (float, optional): Timestep to use in numerical integration. Defaults to 0.2.
            debug (bool, optiona): If True the results of each iteration are printed. If False, nothing is printed.

        """
        test_engine = self
        At = test_engine.nozzle.At
        bounds = np.array([1, 100])*At        #Hardcoded area ratio limits

        #We need to calculate bounds to avoid causing flow seperation in the nozzle
        Ae_for_sepeation = self.seperation_Ae(ambiance.Atmosphere(0).pressure[0])    #Exit area that would cause seperation at sea level.
        if Ae_for_sepeation < bounds[1]:
            bounds[1] = Ae_for_sepeation

        #Function to calculate apogee given the nozzle exit area Ae
        def get_apogee(Ae):
            #Throat area is fixed, we're simply varying the exit area
            test_engine.nozzle = Nozzle(At, Ae, test_engine.nozzle.type, test_engine.nozzle.length_fraction)
            apogee_estimate = estimate_apogee(dry_mass, propellant_mass, test_engine, cross_sectional_area, drag_coefficient, dt)

            if debug == True:
                print(f"Area ratio = {Ae/At}, apogee = {apogee_estimate/1000} km")

            return apogee_estimate

        def func_to_minimise(Ae):
            return -get_apogee(Ae)  #Just make it the negative of the apogee - so we want to minimise the output of this function (scipy can only minimise, not maximise)

        if debug==True:
            print(f"Starting optimisation with bounds Ae/At = {bounds/At}")

        optimum_Ae = scipy.optimize.minimize_scalar(func_to_minimise, bounds = bounds, method="bounded").x
        max_apogee = get_apogee(optimum_Ae)
        self.nozzle = Nozzle(At, optimum_Ae, self.nozzle.type, self.nozzle.length_fraction)

        print(f"Area ratio optimised with Ae/At = {self.nozzle.Ae/self.nozzle.At}, giving apogee = {max_apogee/1000} km")

