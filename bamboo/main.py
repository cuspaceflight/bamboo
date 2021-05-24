'''
Module for calculating nozzle geometry from the chamber conditions. All units SI unless otherwise stated. All angles in radians unless otherwise stated.

Assumptions:
    - 1D flow.
    - Isentropic flow.
    - Perfect gases.

Conventions:
    - Position (x) along the nozzle is defined by: x = 0 at the throat, x < 0 in the combustion chamber, x > 0 in the diverging section of the nozzle.

Known issues:
    - A hardcoded fix is in place for using area ratios outside the Rao angle data range (it tricks the code into making something close to a 15 degree cone). A more robust fix would be better.
    - h_gas_model = '2' doesn't seem to work very well (if at all) right now.

Room for improvement:
    - Should check if the Engine.channel_geometry() function is working as intended.
    - Unsure if the first step (i = 0) in Engine.steady_heating_analysis() is dealt with correctly when using h_gas_model == '3'.
    - Rao bell nozzle data is currently obtained rather crudely (by using an image-of-graph-to-data converter). Would be nicer to have more exact data values.
    - Cone nozzles are not currently implemented.

Subscripts:
    - 0 - Stagnation condition
    - c - Chamber condition (should be the same as stagnation conditions)
    - t - At the throat
    - e - At the nozzle exit plane
    - amb - Atmopsheric/ambient condition

References:
    - [1] - The Thrust Optimised Parabolic nozzle, AspireSpace, http://www.aspirespace.org.uk/downloads/Thrust%20optimised%20parabolic%20nozzle.pdf   \n
    - [2] - Rocket Propulsion Elements, 7th Edition  \n
    - [3] - Design and analysis of contour bell nozzle and comparison with dual bell nozzle https://core.ac.uk/download/pdf/154060575.pdf 
    - [4] - Modelling ablative and regenerative cooling systems for an ethylene/ethane/nitrous oxide liquid fuel rocket engine, Elizabeth C. Browne, https://mountainscholar.org/bitstream/handle/10217/212046/Browne_colostate_0053N_16196.pdf?sequence=1&isAllowed=y  \n
    - [5] - Thermofluids databook, CUED, http://www-mdp.eng.cam.ac.uk/web/library/enginfo/cueddatabooks/thermofluids.pdf    \n
    - [6] - A Simple Equation for Rapid Estimation of Rocket Nozzle Convective Heat Transfer Coefficients, Dr. R. Bartz, https://arc.aiaa.org/doi/pdf/10.2514/8.12572
    - [7] - Regenerative cooling of liquid rocket engine thrust chambers, ASI, https://www.researchgate.net/profile/Marco-Pizzarelli/publication/321314974_Regenerative_cooling_of_liquid_rocket_engine_thrust_chambers/links/5e5ecd824585152ce804e244/Regenerative-cooling-of-liquid-rocket-engine-thrust-chambers.pdf  \n
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import ambiance
import bamboo.cooling as cool
import json
import matplotlib.patches
import thermo.mixture

R_BAR = 8.3144621e3         #Universal gas constant (J/K/kmol)
g0 = 9.80665                #Standard gravitational acceleration (m/s^2)


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



def estimate_apogee(dry_mass, propellant_mass, engine, cross_sectional_area, drag_coefficient = 0.75, dt = 0.2, show_plot = False):
    """Gets an estimate of the apogee reached by a rocket using an Engine object for propulsion.

    Args:
        dry_mass (float): Dry mass of the rocket (kg).
        propellant_mass (float): Propellant mass of the rocket (kg)
        engine (Engine): Engine object to use for propulsion
        cross_sectional_area (float): Cross sectional area used to normalise the drag coefficient (m^2).
        drag_coefficient (float, optional): Approximate drag coefficient for the rocket. Defaults to 0.75.
        dt (float, optional): Timestep size to use. Defaults to 0.2.
        show_plot (bool, optional): Whether or not to plot data at the end. Defaults to False.

    Returns:
        float: Apogee reached by the rocket (m).
    """
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
                raise ValueError(f"Flow separation occured in the nozzle at an altitude of {fn[0]/1000} km")
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
        raise ValueError(f"The area ratio provided ({area_ratio}) is outside of the range of available data. Maximum available is {data['area_ratio'][-1]}, minimum is {data['area_ratio'][0]}.")
    
    else:
        #Linearly interpolate and return the result, after converting it to radians.
        return np.interp(area_ratio, data["area_ratio"], data["theta_n"]) * np.pi/180

def rao_theta_e(area_ratio, length_fraction = 0.8):
    """Returns the contour angle at the exit of the bell nozzle, by interpolating data.  
    Data obtained by using http://www.graphreader.com/ on the graph in Reference [1].

    Note:
        Not actually used by Bamboo (a quadratic only has three degrees of freedom, so Bamboo fits the nozzle contour quadratic to the inflection position, inflection angle, and exit position).

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
        raise ValueError(f"The area ratio provided ({area_ratio}) is outside of the range of available data. Maximum available is {data['area_ratio'][-1]}, minimum is {data['area_ratio'][0]}.")
    
    else:
        #Linearly interpolate and return the result, after converting it to radians
        return np.interp(area_ratio, data["area_ratio"], data["theta_e"]) * np.pi/180

def get_throat_area(perfect_gas, chamber_conditions):
    """Get the nozzle throat area, given the gas properties and combustion chamber conditions. Assumes perfect gas with isentropic flow.

    Args:
        perfect_gas (PerfectGas): Exhaust gas leaving the combustion chamber.
        chamber_conditions (CombustionChamber): Combustion chamber.

    Returns:
        float: Throat area (m^2)
    """
    return (chamber_conditions.mdot * (perfect_gas.cp*chamber_conditions.T0)**0.5 )/( m_bar(1, perfect_gas.gamma) * chamber_conditions.p0) 

def get_exit_area(perfect_gas, chamber_conditions, p_amb):
    """Get the nozzle exit area, given the gas properties and combustion chamber conditions. Assumes perfect gas with isentropic flow.

    Args:
        perfect_gas (PerfectGas): Gas object.
        chamber_conditions (CombustionChamber): CombustionChamber object
        p_amb (float): Ambient pressure (Pa)

    Returns:
        float: Optimum nozzle exit area (Pa)
    """

    Me = M_from_p(p_amb, chamber_conditions.p0, perfect_gas.gamma)
    return (chamber_conditions.mdot * (perfect_gas.cp*chamber_conditions.T0)**0.5 )/(m_bar(Me, perfect_gas.gamma) * chamber_conditions.p0)


class PerfectGas:
    """Object to store exhaust gas properties. Assumes a perfect gas (ideal gas with constant cp, cv and gamma). Only two properties need to be specified.

    Keyword Args:
        gamma (float): Ratio of specific heats cp/cv.
        cp (float): Specific heat capacity at constant pressure (J/kg/K)
        molecular_weight (float): Molecular weight of the gas (kg/kmol)

    Attributes:
        gamma (float): Ratio of specific heats cp/cv.
        cp (float): Specific heat capacity at constant pressure (J/kg/K)
        molecular_weight (float): Molecular weight of the gas (kg/kmol)
        R (float): Specific gas constant (J/kg/K)
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
            raise ValueError(f"Not enough inputs provided to fully define the Gas. You must provide exactly 2, but you provided {len(kwargs)}.")

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
        type (str, optional): Desired shape, can be "rao", "cone" or "custom". Defaults to "rao".

    Keyword Args:
        At (float): Throat area (m^2) - required for 'rao' and 'cone' nozzles.
        Ae (float): Exit plane area (m^2) - required for 'rao' and 'cone' nozzles.
        length_fraction (float, optional): Length fraction for 'rao' nozzle - used if type = "rao". Defaults to 0.8.
        cone_angle (float, optional): Cone angle (deg) - used for 'cone' nozzles. Defaults to 15.
        xs (list) : List of x positions that the data in the 'ys' argument corresponds to (m) - used for 'custom' nozzles.
        ys (list) : List of y positions corresponding to your custom nozzle contour (m) - used for 'custom' nozzles

    Note:
        If using custom data, your x-array must start at zero, and the smallest y-value must be at x = 0. Bamboo uses the convention that x = 0 is the throat location.

    Attributes:
        length (float): Length of the diverging section (distance between throat and nozzle exit) (m).
        At (float): Throat area (m^2)
        Ae (float): Exit area (m^2)
        Rt (float): Throat radius (m)
        Re (float): Exit radius (m)
    """
    def __init__(self, type = "rao", **kwargs):
    
        if type != "rao" and type != "cone" and type != "custom":
            raise ValueError(f"Nozzle type '{type}' is not currently implemented. Try 'rao', 'cone', or 'custom'")

        self.type = type

        if self.type == "custom":
            try:
                self.x_data = kwargs["xs"]
                self.y_data = kwargs["ys"]
            except KeyError:
                raise KeyError("You must specify both  'xs' and 'ys' when using nozzle type = 'custom'")

            assert self.x_data[0] == 0.0 and min(self.x_data) == 0, "x[0] for type = 'custom' must be equal to zero, and there must be no negative values anywhere."
            assert self.y_data[0] == min(self.y_data), "Smallest value in the y-array must be at index 0, for type = 'custom'. Bamboo uses the convention that throats are at x = 0."

            self.length = max(self.x_data)
            self.Rt = self.y_data[list(self.x_data).index(0.0)]
            self.Re = self.y_data[list(self.x_data).index(max(self.x_data))]
            self.At = np.pi*self.Rt**2
            self.Ae = np.pi*self.Re**2

        else:
            try:
                self.At = kwargs["At"]
                self.Ae = kwargs["Ae"]
            except KeyError:
                raise KeyError("You must specify both 'At' and 'Ae' if not using nozzle type = 'custom.")

            self.Rt = (self.At/np.pi)**0.5   #Throat radius (m)
            self.Re = (self.Ae/np.pi)**0.5   #Exit radius (m)

        if self.type == "cone":
            if "cone_angle" in kwargs:
                self.cone_angle = kwargs["cone_angle"]
            else:
                self.cone_angle = 15
            self.dydx = np.tan(self.cone_angle*np.pi/180)
            self.length = (self.Re - self.Rt)/self.dydx

        elif self.type == "rao":
            if "length_fraction" in kwargs:
                self.length_fraction = kwargs["length_fraction"]
            else:
                self.length_fraction = 0.8

            #Our Rao bell nozzle data is taken from a graph, and data is unavailable below an area ratio of 3.7 or above an area ratio of 47.
            if self.Ae/self.At < 3.7 or self.Ae/self.At > 47:
                print("NOTE: Area ratio is outside of data range for Rao bell nozzle graphs (minimum 3.7, maximum 47). Using a 15 deg cone nozzle instead.")
                self.theta_n = 15*np.pi/180
                self.theta_e_graph = float("NaN")

                self.x_n = 0.382*self.Rt*np.cos(self.theta_n - np.pi/2)                              #Inflection point x-value
                self.y_n = 0.382*self.Rt*np.sin(self.theta_n - np.pi/2) + 0.382*self.Rt + self.Rt    #Inflection point y-value
                self.y_e = self.Re                                                                   #Exit y-value (same as Re)

                #Quadratic is normally: x = ay^2 + by + c, but we set a = 0 to make it a linear cone nozzle.
                self.a = 0

                A = np.array([[1,        0],
                              [self.y_n, 1]], dtype='float')
                
                b = np.array([1/np.tan(self.theta_n), self.x_n], dtype='float')

                self.b, self.c = np.linalg.solve(A, b)

                self.x_e = self.b*self.y_e + self.c
                self.length = self.x_e

            else:
                self.theta_n = rao_theta_n(self.Ae/self.At)             #Inflection angle (rad), as defined in [1]
                self.theta_e_graph = rao_theta_e(self.Ae/self.At)       #Exit angle (rad) read off the Rao graph, as defined in [1] - not actually used in bamboo for the quadratic fitting (just here for reference)

                #Page 5 of Reference [1]:
                self.x_n = 0.382*self.Rt*np.cos(self.theta_n - np.pi/2)                              #Inflection point x-value
                self.y_n = 0.382*self.Rt*np.sin(self.theta_n - np.pi/2) + 0.382*self.Rt + self.Rt    #Inflection point y-value
                self.x_e = 0.8*((self.Re/self.Rt) - 1)*self.Rt/np.tan(np.pi/12)                      #Exit x-value - corresponds to 80% the length of a 15 deg cone I believe
                self.y_e = self.Re                                                                   #Exit y-value (same as Re)
                self.length = self.x_e                                                               #Nozzle length

                #Similar to page 2 of Reference [3]. Set up the matrix problem to get the coefficients for x = ay^2 + by + c
                #We will fit the quadratic using the inflection point and exit coordinates, and the inflection gradient. The exit gradient is ignored.
                A = np.array([[2*self.y_n, 1, 0],
                            [self.y_n**2, self.y_n, 1],
                            [self.y_e**2, self.y_e, 1]], dtype='float')
                
                b = np.array([1/np.tan(self.theta_n), self.x_n, self.x_e], dtype='float')

                self.a, self.b, self.c = np.linalg.solve(A, b)

            self.theta_e = np.arctan2(1, 2*self.a*self.x_e + self.b)

    def __repr__(self):
        if self.type == "rao":
            return f"Rao type nozzle (length fraction = {self.length_fraction}). \nLength = {self.length} m \nThroat area = {self.At} m^2 \nExit area = {self.Ae} m^2 \nArea ratio = {self.Ae/self.At} \nRao inflection angle = {self.theta_n*180/np.pi} deg \nRao exit angle = {self.theta_e*180/np.pi} deg from bamboo ({self.theta_e_graph*180/np.pi} deg from Rao graphs) "
        elif self.type == "cone":
            return f"Cone type nozzle (Cone angle = {self.cone_angle} deg \nLength = {self.length} m \nThroat area = {self.At} m^2 \nExit area = {self.Ae} m^2 \nArea ratio = {self.Ae/self.At} \n"
        elif self.type == "custom":
            return f"Custom nozzle shape (Length = {self.length} m \nThroat area = {self.At} m^2 \nExit area = {self.Ae} m^2 \nArea ratio = {self.Ae/self.At} \n"

    def y(self, x):
        """Returns the distance between the nozzle contour and the centreline, given the axial distance 'x' downstream from the throat. Based on Reference [1] page 5.

        Args:
            x (float): Distance along the centreline from the throat (m)

        Returns:
            float: Distance between the nozzle centreline and the contour (m)
        """
        if x < 0:
            raise ValueError(f"x must be greater than zero. You tried to input {x}.")

        if self.type == "custom":
            return np.interp(x, self.x_data, self.y_data)

        elif self.type == "rao" and x <= self.length:
            #Circular throat section
            if x < self.x_n:
                theta = -np.arccos(x/(0.382*self.Rt)) #Take the negative, because we want an answer in the range [-90 to 0 deg], but numpy gives us the one in the range [0 to 180 deg]
                return 0.382*self.Rt*np.sin(theta) + 0.382*self.Rt + self.Rt
            
            #Parabolic section.
            else:
                if self.a == 0: #This occurs if the area ratio is outside the Rao bell nozzle graph range.
                    #x = by + c
                    return (x-self.c)/self.b
                else:
                    #x = ay^2 + by + c
                    return ((4*self.a*(x-self.c) + self.b**2)**0.5 - self.b)/(2*self.a)   #Rearranging the quadratic on page 2 of Reference [3] to solve for y

        elif self.type == "cone" and x <= self.length:
            return self.Rt + self.dydx*x

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
        """Plots the nozzle geometry. Note that to see the plot, you will need to run matplotlib.pyplot.show().

        Args:
            number_of_points (int, optional): Numbers of discrete points to plot. Defaults to 1000.
        """
        x = np.linspace(0, self.length, number_of_points)
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

    @staticmethod
    def from_engine_components(perfect_gas, chamber_conditions, p_amb, type = "rao", length_fraction = 0.8):
        """Generate nozzle based on given gas properties and combustion chamber conditions

        Args:
            perfect_gas (PerfectGas): PerfectGas object for the exhaust gases.
            chamber_conditions (CombustionChamber): CombustionChamber object.
            p_amb (float): Ambient pressure (Pa). The nozzle will be designed to have this pressure at exit.
            type (str, optional): Nozzle type. Can be "rao" or "conical". Conical is not currently implemented. Defaults to "rao".
            length_fraction (float, optional): Rao nozzle length fraction, as defined in Reference [1]. Defaults to 0.8.

        Returns:
            [Nozzle]: The nozzle object.
        """
        return Nozzle(At = get_throat_area(perfect_gas, chamber_conditions), 
                    Ae = get_exit_area(perfect_gas, chamber_conditions, p_amb), 
                    type = type, length_fraction = length_fraction)

class EngineGeometry:
    """Container for additional engine geometry parameters (mostly chamber geometry). Generally only used internally.

    Using the 'inner_wall_thickness' (or 'outer_wall_thickness', if also provided) argument:
        If an array, must be thicknesses at equally spaced x positions. This will be stretched to fill the engine length.
        E.g. [1e-3, 5e-3] will have 1mm thick walls at chamber entrance, 5mm thick at nozzle exit.

    Note:
        If using style = "custom", only specify the geometry up to the throat - everything downstream of the throat is specified by the Nozzle object. 
        Keep in mind that Bamboo uses the convection that x = 0 at the throat (so all your x value should be negative).

    Args:
        nozzle (float): Nozzle of the engine.
        inner_wall_thickness (float or array): Thickness of the inner liner wall (m). Can be constant (float), or variable (array). 
        style (str, optional): Geometry style to use, can be "auto" or "custom". Defaults to "auto".

    Keyword Args:
        chamber_length (float): Length of the combustion chamber (m)
        chamber_area (float): Cross sectional area of the combustion chamber (m^2)
        outer_wall_thickness (float or array): Thickness of the outer liner wall (m). Can be constant (float), or variable (array).
        xs (list) : x-array corresponding to x positions (m) for the 'ys' keyword argument - only used with style = "custom".
        ys (list): y positions specifying the engine contour (m) - only used with style = "custom."

    Attributes:
        x_min (float): Minimum x position (m).
        x_max (float): Maximum x position (m).
        x_chamber_end (float): x position where the combustion chamber ends and the converging section of the nozzle begins (m) - only available of style = "auto".
        x_curved_converging_start (float): x position where the curved part of the converging section begins (m) - only available of style = "auto" and nozzle type = "rao".
        chamber_length (float): Chamber length (m) - only available if style = "auto".
        chamber_area (float): Chamber area (m^2) - only available if style = "auto".
        chamber_radius (float): Chamber radius (m) - only available if style = "auto". 

    """
    def __init__(self, nozzle, inner_wall_thickness, style = "auto", **kwargs):
        self.style = style

        self.x_max = nozzle.length

        if type(inner_wall_thickness) is float or type(inner_wall_thickness) is int:
            self.inner_wall_thickness = [inner_wall_thickness]  #Convert into a list so the interpolation works
        else:
            self.inner_wall_thickness = inner_wall_thickness
        
        if "outer_wall_thickness" in kwargs:
            if type(kwargs["outer_wall_thickness"]) is float or type(kwargs["outer_wall_thickness"]) is int:
                self.outer_wall_thickness = [kwargs["outer_wall_thickness"]]  #Convert into a list so the interpolation works
            else:
                self.outer_wall_thickness = kwargs["outer_wall_thickness"]

        if self.style == "custom":
            try:
                self.x_data = kwargs["xs"]
                self.y_data = kwargs["ys"]
            except KeyError:
                raise KeyError("You must specify both 'xs' and 'ys' when using geometry style = 'custom'")

            assert self.x_data[-1] == 0, "x[-1] must be equal to zero - this datapoint corresponds to the throat."
            assert self.y_data[-1] == nozzle.Rt, "Discontinuity at the throat, y[-1] must the same as the"

            self.x_min = self.x_data[0]

        elif self.style == "auto":
            try:
                self.chamber_length = kwargs["chamber_length"]
                self.chamber_area = kwargs["chamber_area"]
            except KeyError:
                raise KeyError("You must specify both 'chamber_length' and 'chamber_area' when using geometry style = 'auto'")

            self.chamber_radius = (self.chamber_area/np.pi)**0.5

            if nozzle.At > self.chamber_area:
                raise ValueError(f"The combustion chamber area {self.chamber_area} m^2 is smaller than the throat area {nozzle.At} m^2.")

            if nozzle.type == "cone":
                self.dydx_conv = np.tan(-45*np.pi/180)
                self.x_chamber_end = (self.chamber_radius - nozzle.Rt)/self.dydx_conv
                self.x_min = self.x_chamber_end - self.chamber_length

            elif nozzle.type == "rao" or nozzle.type == "custom":
                #Use the system defined in Reference [1] - mostly using Eqns (4)
                #Make sure we cap the size of the converging section to the radius of the combustion chamber.
                chamber_radius = (self.chamber_area/np.pi)**0.5
                theta_min = -np.pi - np.arcsin((chamber_radius - nozzle.Rt - 1.5*nozzle.Rt) / (1.5*nozzle.Rt)) 
                if theta_min > -3*np.pi/4:
                    self.theta_curved_converging_start = theta_min
                else:
                    self.theta_curved_converging_start = -3*np.pi/4

                #Find key properties for the converging section
                self.x_curved_converging_start = 1.5*nozzle.Rt*np.cos(self.theta_curved_converging_start)
                self.y_curved_converging_start = 1.5*nozzle.Rt*np.sin(self.theta_curved_converging_start) + 1.5*nozzle.Rt + nozzle.Rt

                #Find the gradient where the curved converging bit starts
                dxdtheta_curved_converging_start = -1.5*nozzle.Rt*np.sin(self.theta_curved_converging_start)
                self.dydx_curved_converging_start = -1.5*nozzle.Rt*np.cos(self.theta_curved_converging_start)/dxdtheta_curved_converging_start

                #Find the x-position where we reach the combustion chamber radius
                self.x_chamber_end = self.x_curved_converging_start - (self.chamber_radius - self.y_curved_converging_start)/self.dydx_curved_converging_start

                #Start and end points of the engine
                self.x_min = self.x_chamber_end - self.chamber_length
            
            else:
                raise ValueError(f"Unrecognisable nozzle type when trying to generate chamber geometry {nozzle.type}")
                

        else:
            raise ValueError("Argument for 'style' must be 'auto' or 'custom'.")

class Engine:
    """Class for representing a liquid rocket engine.

    Args:
        gas (PerfectGas): Gas representing the exhaust gas for the engine.
        chamber_conditions (CombustionChamber): CombustionChamber for the engine.
        nozzle (Nozzle): Nozzle for the engine.

    Attributes:
        c_star (float): C* for the engine (m/s).
        geometry (EngineGeometry): EngineGeometry object (if added).
    """
    def __init__(self, perfect_gas, chamber_conditions, nozzle):
        self.perfect_gas = perfect_gas
        self.chamber_conditions = chamber_conditions
        self.nozzle = nozzle

        #Extra attributes
        self.c_star = self.chamber_conditions.p0 * self.nozzle.At / self.chamber_conditions.mdot
        self.has_exhaust_transport = False
        self.has_cooling_jacket = False
        self.has_ablative = False
        self.has_exhaust_transport = False

        #Check if the nozzle is choked
        required_throat_area = get_throat_area(perfect_gas, chamber_conditions)
        required_mdot = m_bar(1, self.perfect_gas.gamma)*self.nozzle.At*self.chamber_conditions.p0/(self.perfect_gas.cp*self.chamber_conditions.T0)**0.5

        if abs(self.nozzle.At - required_throat_area) > 1e-5:
            raise ValueError(f"""The nozzle throat area is incompatible with the specified chamber conditions. 
            The required throat area is {required_throat_area} m^2, or the required mass flow rate is {required_mdot} kg/s""")

    #Engine geometry functions
    def y(self, x, up_to = 'contour'):
        """Get y position up to a specified part of the engine (e.g. inner contour, inner or outer side of the ablative, inner or outer side of the inner liner).

        Args:
            x (float): x position (m). x = 0 is the throat, x > 0 is the nozzle diverging section.
            up_to (str): The engine component you want the radius up to. Options include 'contour', 'ablative in', 'ablative out', 'wall in', 'wall out'. Defaults to 'contour'.

        Returns:
            float: Radius up to the given component (m).
        """
        if up_to == 'contour':
            #In the diverging section of the nozzle
            if x >= 0:
                return self.nozzle.y(x)

            #If x is beyond the front of the engine
            elif x < self.geometry.x_min:
                raise ValueError(f"x is beyond the front of the engine. You tried to input {x} but the minimum value you're allowed is {self.geometry.x_min}")

            #Left hand side of throat
            else:
                try:
                    self.geometry
                except AttributeError:
                    raise AttributeError("Geometry is not defined for x < 0. You need to add geometry with the 'Engine.add_geometry()' function.")

                #Custom geometry
                if self.geometry.style == "custom":
                    return np.interp(x, self.geometry.x_data, self.geometry.y_data)

                elif self.geometry.style == "auto":
                    #Rao bell nozzle
                    if self.nozzle.type == "rao" or self.nozzle.type == "custom":
                        #Curved converging section
                        if x < 0 and x > self.geometry.x_curved_converging_start:
                            theta = -np.arccos(x/(1.5*self.nozzle.Rt))
                            return 1.5*self.nozzle.Rt*np.sin(theta) + 1.5*self.nozzle.Rt + self.nozzle.Rt

                        #Before the curved part of the converging section
                        elif x <= self.geometry.x_curved_converging_start:

                            #Inside the chamber
                            if x < self.geometry.x_chamber_end and x >= self.geometry.x_min:
                                return self.geometry.chamber_radius

                            #Inside the converging section
                            elif x >= self.geometry.x_chamber_end:
                                return np.interp(x, 
                                                [self.geometry.x_chamber_end, self.geometry.x_curved_converging_start], 
                                                [self.geometry.chamber_radius, self.geometry.y_curved_converging_start])

                    #Cone nozzle
                    elif self.nozzle.type == "cone":
                        #If between the end of the chamber and the throat
                        if x < 0 and x >= self.geometry.x_chamber_end:
                            #Use a 45 degree converging section
                            return self.nozzle.Rt + self.geometry.dydx_conv*x

                        #If in the chamber
                        elif x >= self.geometry.x_min and x < self.geometry.x_chamber_end:
                            return self.geometry.chamber_radius

        elif up_to == 'ablative in':
            if self.has_ablative == False:
                raise AttributeError("There is no ablative attached to this engine")
            else:
                return self.y(x)
        
        elif up_to == 'ablative out':
            if self.has_ablative == False:
                raise AttributeError("There is no ablative attached to this engine")
            else:
                return self.y(x) + self.thickness(x, layer = 'ablative')
        
        elif up_to == 'wall in':
            if self.has_ablative:
                return self.y(x, up_to = 'ablative out')
            else:
                return self.y(x)
        
        elif up_to == 'wall out':
            if self.has_ablative:
                return self.y(x, up_to = 'ablative out') + self.thickness(x, layer = 'wall')
            else:
                return self.y(x) + self.thickness(x, layer = 'wall')

        else:
            raise ValueError(f"'{up_to}' is not a valid part of the engine. Try 'contour', 'ablative in', 'ablative out', 'wall in' or 'wall out'")

    def A(self, x):
        """Get the engine cross sectional area at a given x position.

        Args:
            x (float): x position (m). x = 0 is the throat, x > 0 is the nozzle diverging section.

        Returns:
            float: Cross sectional area (m^2)
        """
        return np.pi*self.y(x)**2

    def thickness(self, x, layer):
        """Get the thickness of the engine wall, or ablative, at a specific point.

        Args:
            x (float): x position (m)
            layer (str): 'ablative' or 'wall'

        Returns:
            float: Thickness at the given value of x (m)
        """
        if layer == 'wall':
            #Stretch the wall_thickness array across the engine, and interpolate to get our desired thickness
            inner_wall_thickness_xs = np.linspace(self.geometry.x_min, self.geometry.x_max, len(self.geometry.inner_wall_thickness))
            return np.interp(x, inner_wall_thickness_xs, self.geometry.inner_wall_thickness)
        
        if layer == 'ablative':
            if self.has_ablative == False:
                raise AttributeError("This engine does not have an ablative attached")

            #Check if there is ablative in this region of the engine
            if self.ablative.xs[0] <= x <= self.ablative.xs[1]:
                if self.ablative.ablative_thickness is None:
                    #If self.ablative_thickness is None, fill up the distance between the nozzle contour and the chamber radius with ablative
                    return self.geometry.chamber_radius - self.y(x)

                else:
                    #Stretch the ablative_thickness array over the range that ablatives are present, and interpolate to get our desired thickness
                    ablative_thickness_xs = np.linspace(self.ablative.xs[0], self.ablative.xs[1], len(self.ablative.ablative_thickness))
                    return np.interp(x, ablative_thickness_xs, self.ablative.ablative_thickness)
            else:
                #If we're outside the region where the ablative is present, return zero thickness.
                return 0.0

    #Thermodynamic properties as a function of position
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
                return self.chamber_conditions.mdot*(self.perfect_gas.cp*self.chamber_conditions.T0)**0.5 / (self.A(x)*self.chamber_conditions.p0) - m_bar(Mach, self.perfect_gas.gamma)
            
            if x > 0:
                Mach = scipy.optimize.root_scalar(func_to_solve, bracket = [1, 500], x0 = 1).root
            else:
                Mach = scipy.optimize.root_scalar(func_to_solve, bracket = [0.0,1], x0 = 0.5).root
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

    def rho(self, x):
        """Get exhaust gas density.

        Args:
            x (float): Axial position. Throat is at x = 0.

        Returns:
            float: Freestream gas density (kg/m^3)
        """
        #p = rhoRT for an ideal gas, so rho = p/RT
        return self.p(x)/(self.T(x)*self.perfect_gas.R)

    
    #Thrust related functions
    def check_separation(self, p_amb):
        """Approximate check for nozzle separation. Based off page 17 of Reference [2].  
        separation occurs when P_wall/P_amb = 0.583 * (P_amb/P_chamber)^(0.195)

        Args:
            p_amb (float): Ambient (atmospheric) pressure.
        
        Returns:
            bool or float: Returns the position x (m) from the throat at which separation occurs, if it does occur. If not, it returns False.
        """
        
        #Get the value of P_wall/P_amb requried for separation
        separation_pressure_ratio = 0.583 * (p_amb/self.chamber_conditions.p0)**0.195

        #separation can't occur if there's a vacuum outside:
        if p_amb == 0:
            return False

        #Check for separation by comparing to the lowest pressure in the nozzle (which will be at exit):
        elif self.p(self.nozzle.length)/p_amb < separation_pressure_ratio:
            def func_to_solve(x):
                return self.p(x)/p_amb - separation_pressure_ratio  #Should equal zero at the separation point
            
            return scipy.optimize.root_scalar(func_to_solve, bracket = [0, self.nozzle.length], x0 = 0).root    #Find the separation position. 
            
        else:
            return False

    def separation_p_amb(self):
        """Approximate way of getting the ambient pressure at which nozzle wall separation occurs. Based off page 17 of Reference [2].  
        separation occurs when P_wall/P_amb = 0.583 * (P_amb/P_chamber)^(0.195). 
        It will first occur when P_e = P_wall satisfies this equation, since P_e is the lowest pressure in the nozzle.

        Returns:
            float: Ambient pressure at which separation first occurs (Pa)
        """
        pc = self.chamber_conditions.p0
        pe = self.p(self.nozzle.length)
        return ((pe*pc**0.195)/0.583)**(1/1.195)

    def separation_Ae(self, p_amb):
        """Approximate way of getting the exit area at which nozzle wall separation occurs. Based off page 17 of Reference [2].  
        separation occurs when P_wall/P_amb = 0.583 * (P_amb/P_chamber)^(0.195). 

        Returns:
            float: Exit area at which separation occurs for the given p_amb (m^2)
        """
        separation_wall_pressure = p_amb * 0.583 * (p_amb/self.chamber_conditions.p0)**0.195

        #Get the exit area that gives this wall pressure at the exit.
        return get_exit_area(self.perfect_gas, self.chamber_conditions, separation_wall_pressure)

    def thrust(self, p_amb):
        """Returns the thrust of the engine for a given ambient pressure.

        Args:
            p_amb (float): Ambient pressure (Pa)

        Returns:
            float: Thrust (N)
        """
        if self.check_separation(p_amb) ==  False:
            Me = self.M(self.nozzle.length)
            Te = self.T(self.nozzle.length)
            pe = self.p(self.nozzle.length)

            return self.chamber_conditions.mdot*Me*(self.perfect_gas.gamma*self.perfect_gas.R*Te)**0.5 + (pe - p_amb)*self.nozzle.Ae    #Generic equation for rocket thrust
        
        else:
            raise ValueError(f"separation occured in the nozzle, at a postion {self.check_separation(p_amb)} m downstream of the throat.")

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

        #We need to calculate bounds to avoid causing flow separation in the nozzle
        Ae_for_sepeation = self.separation_Ae(ambiance.Atmosphere(0).pressure[0])    #Exit area that would cause separation at sea level.
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


    #Plotting functions
    def plot_geometry(self, number_of_points = 1000, minimal = False, legend = True):
        """Plots the engine geometry. Note that all spiral cooling jacket geometry is shown with equally spaced rectangles, 
        even if irregularly spaced non-rectangular shapes are used. The rectangles have a width equal to the channel effective diameter,
        and an area equal to the channel flow area.
        To see the plot, you will need to run matplotlib.pyplot.show().

        Args:
            number_of_points (int, optional): Numbers of discrete points to plot. Defaults to 1000.
            minimal (bool, optional): If True, the engine contour is plotted as a single line. If False, the line's thickness will vary to show wall thickness, and other geometry details will be shown. Defaults to False.
            legend (bool, optional): If True a legend is shown. If False, it isn't. Defaults to True.
        """
        try:
            self.geometry
        except AttributeError:
            raise AttributeError("Geometry has not been added, so can't run Engine.plot_geometry(). You need to add geometry with the 'Engine.add_geometry()' function.")
        
        x = np.linspace(self.geometry.x_min, self.geometry.x_max, number_of_points)

        fig, axs = plt.subplots()

        #Minimalistic plotting - only show the engine contour, without any extra features
        if minimal:
            y = np.zeros(len(x))
            for i in range(len(x)):
                y[i] = self.y(x[i])

            axs.plot(x, y, color="blue")
            axs.plot(x, -y, color="blue")

        #Normal plotting - show any wall thickness to scale, display any ablatives, and show a representation of the cooling jacket
        else:
            #Initialise arrays
            if self.has_ablative:
                ablative_inner = np.zeros(len(x))
                ablative_outer = np.zeros(len(x))

            wall_inner = np.zeros(len(x))
            wall_outer = np.zeros(len(x))
            
            for i in range(len(x)):
                if self.has_ablative:
                    #Get the ablative y values at each x
                    ablative_inner[i] = self.y(x[i], up_to = 'ablative in')
                    ablative_outer[i] = self.y(x[i], up_to = 'ablative out')

                #Get the wall y values at each x
                wall_inner[i] = self.y(x[i], up_to = 'wall in')
                wall_outer[i] = self.y(x[i], up_to = 'wall out')

            if self.has_ablative:
                #Plot the ablative to scale
                axs.fill_between(x, ablative_inner, ablative_outer, color="grey", label = 'Ablative')
                axs.fill_between(x, -ablative_inner, -ablative_outer, color="grey")
                
            #Plot the engine wall thickness to scale
            axs.fill_between(x, wall_inner, wall_outer, color="blue", label = 'Wall')
            axs.fill_between(x, -wall_inner, -wall_outer, color="blue")

            #Show the cooling jacket - is a bit rough right now
            if self.has_cooling_jacket:
                #Range of xs that the jacket is between
                if self.geometry.x_min > self.cooling_jacket.xs[0]:
                    xmin = self.geometry.x_min
                else:
                    xmin = self.cooling_jacket.xs[0]

                if self.geometry.x_max < self.cooling_jacket.xs[1]:
                    xmax = self.geometry.x_max
                else:
                    xmax = self.cooling_jacket.xs[1]

                #If using a spiral cooling jacket
                if self.cooling_jacket.configuration == 'spiral':

                    #Just for the legends
                    axs.plot(0, 0, color = 'green', label = 'Cooling channels')  
                    if self.cooling_jacket.number_of_ribs != 1:
                        axs.plot(0, 0, color = 'red', label = 'Channel ribs')  
                        rib_color = 'red'
                    else:
                        rib_color = 'green'

                    #Plot the spiral channels as rectangles
                    current_x = self.geometry.x_min

                    while current_x < self.geometry.x_max:
                        y_jacket_inner = self.y(current_x, up_to = "wall out")   #y position of inner side of cooling channel wall
                        H = self.cooling_jacket.channel_height(current_x)          #Current channel height
                        W = self.cooling_jacket.channel_width(current_x)           #Current channel width

                        #Show the ribs as filled in rectangles
                        area_per_rib = W*H*self.cooling_jacket.blockage_ratio/self.cooling_jacket.number_of_ribs
                        rib_width = area_per_rib/H

                        for j in range(self.cooling_jacket.number_of_ribs):
                            distance_to_next_rib = W/self.cooling_jacket.number_of_ribs

                            #Make all ribs red
                            axs.add_patch(matplotlib.patches.Rectangle([current_x + j*distance_to_next_rib, y_jacket_inner], rib_width, H, color = rib_color, fill = True))
                            axs.add_patch(matplotlib.patches.Rectangle([current_x + j*distance_to_next_rib, -y_jacket_inner-H], rib_width, H, color = rib_color, fill = True))

                        #Plot 'outer' cooling channel (i.e. the amount moved per spiral)
                        axs.add_patch(matplotlib.patches.Rectangle([current_x, y_jacket_inner], W, H, color = 'green', fill = False))
                        axs.add_patch(matplotlib.patches.Rectangle([current_x, -y_jacket_inner-H], W, H, color = 'green', fill = False))

                        current_x = current_x + W

                #If using a vertical channels cooling jacket
                if self.cooling_jacket.configuration == 'vertical':
                    regen_xs = np.linspace(xmin, xmax, 1000)
                    channel_inner_mapped = np.interp(regen_xs, x, wall_outer)
                    channel_height_mapped = np.zeros(len(regen_xs))

                    for i in range(len(regen_xs)):
                        channel_height_mapped[i] = self.cooling_jacket.channel_height(regen_xs[i])

                    #Show the channel thickness to scale
                    axs.fill_between(regen_xs, channel_inner_mapped, channel_inner_mapped+channel_height_mapped, color="green", label = 'Cooling channel')
                    axs.fill_between(regen_xs, -channel_inner_mapped, -channel_inner_mapped-channel_height_mapped, color="green")
            
            if legend:
                axs.legend()

        axs.grid()
        axs.set_aspect('equal')
        plt.xlabel("x position (m)")
        plt.ylabel("y position (m)")

    def plot_gas_temperature(self, number_of_points=1000):
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

    def plot_gas_mach(self, number_of_points=1000):
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


    #Adding additional components and specifications to the engine
    def add_geometry(self, inner_wall_thickness, style="auto", **kwargs):
        """Specify extra geometry parameters. Required for running cooling system and stress analyses.

        Using the 'inner_wall_thickness' (or 'outer_wall_thickness', if also provided) argument:
            If an array, must be thicknesses at equally spaced x positions. This will be stretched to fill the engine length.
            E.g. [1e-3, 5e-3] will have 1mm thick walls at chamber entrance, 5mm thick at nozzle exit.

        Note:
            If using style = "custom", only specify the geometry up to the throat - everything downstream of the throat is specified by the Nozzle object. 
            Keep in mind that Bamboo uses the convection that x = 0 at the throat (so all your x values should be negative).

        Args:
            inner_wall_thickness (float or array): Thickness of the inner liner wall (m). Can be constant (float), or variable (array). 
            style (str, optional): Geometry style to use, can be "auto" or "custom". Defaults to "auto".

        Keyword Args:
            chamber_length (float): Length of the combustion chamber (m) - needed for style = 'auto'.
            chamber_area (float): Cross sectional area of the combustion chamber (m^2) - needed for style = 'auto'.
            xs (list) : x-array corresponding to x positions (m) for the 'ys' argument - needed for style = 'custom'.
            ys (list): y positions specifying the engine contour (m) - needed for style = 'custom.
            outer_wall_thickness (float or array, optional): Thickness of the outer liner wall (m). Can be constant (float), or variable (array). Used for stress analyses.
        """

        self.geometry = EngineGeometry(self.nozzle, inner_wall_thickness, style, **kwargs)
        self.has_geometry = True

    def add_cooling_jacket(self, inner_wall_material, inlet_T, inlet_p0, coolant_transport, mdot_coolant, xs = None, configuration = "spiral", **kwargs):
        """Container for cooling jacket information - e.g. for regenerative cooling.

        Args:
            inner_wall_material (Material): Wall material on the inner side of the cooling jacket.
            inlet_T (float): Inlet coolant temperature (K)
            inlet_p0 (float): Inlet coolant stagnation pressure (Pa)
            coolant_transport (TransportProperties): Container for the coolant transport properties.
            mdot_coolant (float): Coolant mass flow rate (kg/s)
            xs (list): x positions that the cooling jacket starts and ends at, [x_min, x_max]. By default will encompass the entire engine.
            configuration (str, optional): Options include 'spiral' and 'vertical'. Defaults to "vertical".
            has_ablative (bool, optional): Whether or not the engine has an ablative.
        
        Keyword Args:
            blockage_ratio (float): This is the proportion (by area) of the channel cross section occupied by ribs.
            number_of_ribs (int): Only relevant if 'blockage_ratio' !=0. This is the number of ribs present in the cooling channel. For spiral channels this is the number of ribs 'per pitch' - it is numerically equal to the number of parallel spiral channels.
            channel_height (float or list): This is the height of the channels, in the radial direction (m). Can be a constant (float) or list. If list, it must be an channel heights at regularly spaced x positions, which will be stretched to fit the xs input.
            channel_width (float): Only relevant if configuration = 'spiral'. This is the total width (i.e. pitch) of the cooling channels (m).
            outer_wall_material (Material): Wall material for the outer liner.
        """
        if self.has_geometry == False:
            raise AttributeError("Need to add geometry to the engine first with Engine.add_geometry() before adding a cooling jacket")

        #Make sure the values in xs aren't outside of the engine
        if xs is None:
            xs = [self.geometry.x_min, self.geometry.x_max]
        elif xs[0] < self.geometry.x_min:
            raise ValueError(f"xs[0] is upstream of the front of the engine (x = {self.geometry.x_min} m)")
        elif xs[1] > self.geometry.x_max:
            raise ValueError(f"xs[1] is downstream of the end of the engine (x = {self.geometry.x_max} m)")
        
        self.cooling_jacket = cool.CoolingJacket(inner_wall_material,
                                                inlet_T, 
                                                inlet_p0, 
                                                coolant_transport, 
                                                mdot_coolant, 
                                                xs, 
                                                configuration, 
                                                **kwargs)
        self.has_cooling_jacket = True

    def add_exhaust_transport(self, transport_properties):
        """Add a model for the exhaust gas transport properties (e.g. viscosity, thermal doncutivity, etc.). This is needed to run cooling system analyses.

        Args:
            transport_properties (TransportProperties): Container for the exhaust gas transport properties.
        """
        self.x_ehaust_transport = transport_properties
        self.has_exhaust_transport = True

    def add_ablative(self, ablative_material, wall_material = None, xs = None, ablative_thickness = None, regression_rate = 0.0):
        """
        Note:
            The wall material you add will override the inner wall material of any cooling jackets that are present.

        Args:
            ablative_material (Material): Ablative material.
            wall_material (Material): Wall material on the outside of the ablative (will override the cooling jacket wall material). Defaults to None, in which case the cooling jacket material will be used.
            xs (list, optional): x positions that the ablative is present between, [xmin, xmax]. By default will encompass the whole engine.
            ablative_thickness (float or list): Thickness of ablative. If a list is given, it must correspond to thickness at regular x intervals, which will be stretched out over the inverval of 'xs'. Defaults to None (in which case the ablative extends from the engine contour to combustion chamber radius).
            regression_rate (float): (Not currently used) (m/s). Defaults to 0.0.
        """


        #Use the cooling jacket's wall material if the user inputs 'wall_material = None'
        if wall_material == None:
            if self.has_cooling_jacket == False:
                raise AttributeError("You need to specify a wall material for the ablative (there is no cooling jacket wall material to use)")
            wall_material_to_use = self.cooling_jacket.inner_wall_material

        else:
            wall_material_to_use = wall_material
        
        if self.has_geometry == False:
            raise AttributeError("Need to add geometry to the engine first with Engine.add_geometry() before adding an ablative")

        #Make sure the values in xs aren't outside of the engine
        if xs is None:
            xs = [self.geometry.x_min, self.geometry.x_max]
        elif xs[0] < self.geometry.x_min:
            raise ValueError(f"xs[0] is upstream of the front of the engine (x = {self.geometry.x_min} m)")
        elif xs[1] > self.geometry.x_max:
            raise ValueError(f"xs[1] is downstream of the end of the engine (x = {self.geometry.x_max} m)")

        self.ablative = cool.Ablative(ablative_material = ablative_material,
                                      wall_material = wall_material_to_use, 
                                      regression_rate = regression_rate, 
                                      xs = xs, 
                                      ablative_thickness = ablative_thickness)
        self.has_ablative = True

        if self.has_cooling_jacket is True:
            self.cooling_jacket.has_ablative = True

    #Cooling system functions
    def map_thickness_profile(self, thickness, number_of_points):
        """Stretches an array of any size so that it has the required 'number_of_points', whilst maintaining the same values at indexes [0] and [-1]. 
        All indexes inbetween are filled in with interpolation.
           
           Args:
                thickness (list or array): Thickness distribution. Must contains thicknesses at equally spaced x positions.
                number_of_points (int): Number of discrete liner positions

           Returns:
                mapped_thickness (array): Interpolated thickness profile
           """
        mapped_thickness = np.zeros(number_of_points)

        for i in range(number_of_points):
            #Fraction of the way along the engine
            pos_fraction = i/(number_of_points-1)

            #Index we want from our thickness array (may be non-integer)
            thickness_index = pos_fraction * (len(thickness)-1)

            #Get the thickness at that index
            mapped_thickness[i] = np.interp(thickness_index, range(len(thickness)), thickness)

        return mapped_thickness

    def coolant_path_length(self, number_of_points = 1000):
        """Finds the path length of the coolant in the jacket from engine geometry and channel configuration.
           Number_of_points input must be equal to number_of_points used in a heating analysis.

        Note:
            Recently changed the system for finding the engine contour - would be useful to run some proper tests on this function to see if the values it returns still make sense.

        Args:
            number_of_points (int, optional): Number of discretised points to split path into. Defaults to 1000.

        Returns:
            array: Discretised coolant path length array with "number_of_sections" elements. (m).
        """
        discretised_x = np.linspace(self.geometry.x_max, self.geometry.x_min, number_of_points)
        dx = discretised_x[0] - discretised_x[1]

        #axis_length = self.geometry.x_max - self.geometry.x_min     # Axial engine length
        discretised_length = np.zeros(number_of_points-1)

        if self.cooling_jacket.configuration == "spiral":
            for i in range(number_of_points-1):     
                pitch = self.cooling_jacket.channel_width(discretised_x[i])     # Pitch can vary - this is the instantaneous pitch

                y_i = self.y(discretised_x[i], up_to = "wall out") + self.cooling_jacket.channel_height(discretised_x[i])/2
                y_iplus1 = self.y(discretised_x[i+1], up_to = "wall out") + self.cooling_jacket.channel_height(discretised_x[i+1])/2

                # Find the average radius for this section and use it to determine the spiral section length
                radius_avg = (y_i + y_iplus1)/2 
                dy = y_iplus1 - y_i

                #[small path length]^2 = [tangential distance moved]^2 + [radial distance moved]^2 + [axial distance moved]^2
                #[dl]^2 = [radius*(angle turned)]^2 + [dy]^2 + [dx]^2
                angle_turned = (dx/pitch) * 2 *np.pi
                dl = ((radius_avg*angle_turned)**2 + dy**2 + dx**2)**(1/2)

                discretised_length[i] = dl

            return discretised_length

        if self.cooling_jacket.configuration == "vertical":
            for i in range(number_of_points-1):

                y_i = self.y(discretised_x[i], up_to = "wall out") + self.cooling_jacket.channel_height(discretised_x[i])/2
                y_iplus1 = self.y(discretised_x[i+1], up_to = "wall out") + self.cooling_jacket.channel_height(discretised_x[i+1])/2

                dy = y_iplus1 - y_i
                discretised_length[i] = (np.sqrt(dy**2 + dx**2))

            return discretised_length

        else:
            raise AttributeError("Invalid cooling channel configuration")

    def coolant_friction_factor(self, T, p, x, y):
        """Determine the friction factor of the coolant at the current position.
           Formula from reference [4] page 29 - seems to be a friction factor equation from Petukhov, described in the 'Fundamentals of Heat and Mass Transfer' textbook.
           Works for turbulent flow in smooth tubes, in the Reynolds number range (3000 < Re_D < 5e6).

        Args:
            x (float): Axial position
            y (float, optional): y distance from engine centreline to the inner wall of the cooling channel (m).
            T (float): Coolant temperature at x            
            p (float): Coolant pressure at x
        Returns:
            float: Dimensionless friction factor
        """
        mu = self.cooling_jacket.coolant_transport.mu(T=T, p=p)
        rho = self.cooling_jacket.coolant_transport.rho(T=T, p=p)
        D = self.cooling_jacket.D(x, y)
        v = self.cooling_jacket.coolant_velocity(rho, x, y)

        reynolds = rho*v*D/mu

        return ((0.79*np.log(reynolds)) - 1.64)**(-2)

    def coolant_dynamic_pressure(self, T, p, x = None, y = None):
        """Determine dynamic pressure of coolant.

        Args:
            x (float, optional): Axial position. Only required for 'vertical' channels.
            y (float, optional): y distance from engine centreline to the inner wall of the cooling channel (m).
            T (float): Coolant temperature at x            
            p (float): Coolant pressure at x

        Returns:
            float: Dynamic pressure of coolant (Pa)
        """

        rho = self.cooling_jacket.coolant_transport.rho(T=T, p=p)
        v = self.cooling_jacket.coolant_velocity(rho, x, y)

        return rho*(v**2)/2    

    def coolant_p0_drop(self, friction_factor, dl, T, p, x = None, y = None):
        """Determine the drop in the Bernoulli constant (stagnation pressure) using the friction factor.
        Args:
            friction_factor (float): Dimensionless friction factor
            x (float, optional): Axial position. Only required for 'vertical' channels.
            y (float, optional): y distance from engine centreline to the inner wall of the cooling channel (m).
            dl (float): Length to evaluate pressure drop over - an increment along the channel, not the engine axis
            T (float): Coolant temperature            
            p (float): Coolant pressure
        Returns:
            float: Stagnation pressure drop (Pa)
        """

        D = self.cooling_jacket.D(x, y)
        Q = self.coolant_dynamic_pressure(T=T, p=p, x=x, y=y)

        return friction_factor*dl*Q/D

    def ablative_thermal_circuit(self, r, h_gas, ablative_material, ablative_thickness, T_gas, T_wall):
        """
        q is per unit length along the nozzle wall (axially) - positive when heat is flowing to the coolant.  

        Args:
            r (float): Radius to the contour of the engine (m)
            h_gas (float): Gas side convective heat transfer coefficient
            ablative_material (Material): Material object for the ablative material, needed for thermal conductivity
            ablative_thickness (float): Thickness of the ablative material (m)
            T_gas (float): Free stream gas temperature (K)
            T_wall (float): Wall temperature (K)

        Returns:
            float, float, float, float: q_dot, R_gas, R_ablative
        """

        r_in = r 
        r_out = r_in + ablative_thickness

        A_in = 2*np.pi*r_in    #Inner area per unit length (i.e. just the inner circumference)

        R_gas = 1/(h_gas*A_in)
        R_ablative = np.log(r_out/r_in)/(2*np.pi*ablative_material.k)

        q_dot = (T_gas - T_wall)/(R_gas + R_ablative)    #Heat flux per unit length

        return q_dot, R_gas, R_ablative

    def steady_heating_analysis(self, number_of_points = 1000, h_gas_model = "bartz-sigma", h_coolant_model = "sieder-tate", to_json = "heating_output.json", **kwargs):
        """Steady state heating analysis. Can be used for regenarative cooling, or combined regenerative and ablative cooling.

        Args:
            number_of_points (int, optional): Number of discrete points to divide the engine into. Defaults to 1000.
            h_gas_model (str, optional): Equation to use for the gas side convective heat transfer coefficients. Options are 'rpe', 'bartz' and 'bartz-sigma'. Defaults to "bartz-sigma".
            h_coolant_model (str, optional): Equation to use for the coolant side convective heat transfer coefficients. Options are 'rpe', 'sieder-tate' and 'dittus-boelter'. Defaults to "sieder-tate".
            to_json (str or bool, optional): Directory to export a .JSON file to, containing simulation results. If False, no .JSON file is saved. Defaults to 'heating_output.json'.
        
        Keyword Args:
            gas_fudge_factor (float, optional): Fudge factor to multiply the gas side thermal resistance by. A factor of ~1.3 can sometimes help results match experimental data better.

        Note:
            See the bamboo.cooling module for details of each h_gas and h_coolant option. Defaults are Bartz (using sigma correlation) for gas side, and Sieder-Tate for coolant side. These are believed to be the most accurate.

        Note:
            Sometimes the wall temperature can be above the boiling point of your coolant, in which case you may get nucleate boiling or other effects, and the Sieder-Tate model may become questionable.


        Returns:
            dict: Results of the simulation. Contains the following dictionary keys: 
                - "x" : x positions corresponding to the rest of the data (m)
                - "T_wall_inner" : Exhaust gas side wall temperature (K)
                - "T_wall_outer" : Coolant side wall temperature (K)
                - "T_coolant" : Coolant temperature (K)
                - "T_gas" : Exhaust gas freestream temperature (K)
                - "q_dot" : Heat transfer rate per unit length (axially along the engine) (W/m)
                - "h_gas" : Convective heat transfer rate for the exhaust gas side
                - "h_coolant" : Convective heat transfer rate for the coolant side
                - "enthalpy_coolant" : Coolant specific enthalpy (J/kg) - will only contain data if enthalpy functions were provided.
                - "boil_off_position" : x position of any coolant boil off. Equal to None if the coolant does not boil.
                - (and some more)
        """

        '''Check if we have everything needed to run the simulation'''
        try:
            self.geometry
        except AttributeError:
            raise AttributeError("Cannot run heating analysis without additional geometry definitions. You need to add geometry with the 'Engine.add_geometry()' function.")

        try:
            self.x_ehaust_transport
        except AttributeError:
            raise AttributeError("Cannot run heating analysis without an exhaust gas transport properties model. You need to add one with the 'Engine.add_exhaust_transport()' function.")
        

        '''Initialise variables and arrays'''
        #Check if user specified enthalpy 
        if hasattr(self.cooling_jacket.coolant_transport, 'given_T_from_enthalpy'):
            use_coolant_enthalpy = True
        else:
            use_coolant_enthalpy = False
            print("NOTE: No enthalpy data provided for the coolant - will use specific heat capacity to predict temperature rises.")
            
        #Keep track of if the coolant pressure drops below chamber pressure.
        too_low_pressure = False

        #Discretisation of the nozzle
        discretised_x = np.linspace(self.geometry.x_max, self.geometry.x_min, number_of_points) #Run from the back end (the nozzle exit) to the front (chamber)
        dx = discretised_x[0] - discretised_x[1]

        #Calculation of coolant channel length per "section"
        channel_length = self.coolant_path_length(number_of_points=number_of_points)     #number_of_sections must be equal to number_of_points

        #Data arrays to return
        T_wall_inner = np.full(len(discretised_x), float('NaN')) #Gas side wall temperature
        T_wall_outer = np.full(len(discretised_x), float('NaN')) #Coolant side wall temperature
        T_gas = np.full(len(discretised_x), float('NaN'))        #Freestream gas temperature
        q_dot = np.full(len(discretised_x), float('NaN'))        #Heat transfer rate per unit length
        h_gas = np.full(len(discretised_x), float('NaN'))       #Gas side convective heat transfer coefficient
        R_gas =  np.full(len(discretised_x), float('NaN'))      #Gas side convective thermal resistance
        R_wall = np.full(len(discretised_x), float('NaN'))      #Wall thermal resistance

        mu_gas = np.full(len(discretised_x), float('NaN'))      #Exhaust gas absolute viscosity
        k_gas = np.full(len(discretised_x), float('NaN'))       #Exhaust gas thermal conductivity
        Pr_gas = np.full(len(discretised_x), float('NaN'))      #Exhaust gas Prandtl number

        #Only relevant if there's a cooling jacket:
        T_coolant = np.full(len(discretised_x), float('NaN'))       #Coolant bulk temperature
        enthalpy_coolant = np.full(len(discretised_x), float('NaN'))#Coolant specific enthalpy (J/kg)
        h_coolant = np.full(len(discretised_x), float('NaN'))       #Cooling side convective heat transfer coefficient
        R_coolant = np.full(len(discretised_x), float('NaN'))       #Coolant side convective thermal resistance
        p_coolant = np.full(len(discretised_x), float('NaN'))       #Coolant static pressure
        p0_coolant = np.full(len(discretised_x), float('NaN'))      #Coolant Bernoulli constant / stagnation pressure
        v_coolant = np.full(len(discretised_x), float('NaN'))       #Coolant bulk velocity
        
        Pr_coolant = np.full(len(discretised_x), float('NaN'))      #Coolant bulk Prandtl number
        mu_coolant = np.full(len(discretised_x), float('NaN'))      #Coolant bulk absolute viscosity
        k_coolant = np.full(len(discretised_x), float('NaN'))       #Coolant bulk thermal conductivity
        cp_coolant = np.full(len(discretised_x), float('NaN'))      #Coolant bulk specific heat capacity
        rho_coolant = np.full(len(discretised_x), float('NaN'))     #Coolant bulk density

        #Only relevant if there's an ablative:
        T_ablative_inner = np.full(len(discretised_x), float('NaN'))    #Ablative inner side temperature
        R_ablative = np.full(len(discretised_x), float('NaN'))          #Ablative thermal resistance

        #Check if the user wants to use a fudge factor on the gas side thermal resistance
        if "gas_fudge_factor" in kwargs:
            gas_fudge_factor = kwargs["gas_fudge_factor"]
        else:
            gas_fudge_factor = 1.0

        '''Main loop'''
        for i in range(len(discretised_x)):
            x = discretised_x[i]
            T_gas[i] = self.T(x)

            #Get exhaust gas transport properties
            mu_gas[i] = self.x_ehaust_transport.mu(T = T_gas[i], p = self.p(x))
            k_gas[i] = self.x_ehaust_transport.k(T = T_gas[i], p = self.p(x))
            Pr_gas[i] = self.x_ehaust_transport.Pr(T = T_gas[i], p = self.p(x))

            if self.has_cooling_jacket and self.cooling_jacket.xs[0] <= x <= self.cooling_jacket.xs[1]:
                #Gas side heat transfer coefficient
                if h_gas_model == "rpe":
                    h_gas[i] = cool.h_gas_rpe(2*self.y(x),
                                            self.M(x),
                                            T_gas[i],
                                            self.rho(x),
                                            self.perfect_gas.gamma,
                                            self.perfect_gas.R,
                                            mu_gas[i],
                                            k_gas[i],
                                            Pr_gas[i])

                elif h_gas_model == "bartz":
                    #We need the previous wall temperature to use h_gas_bartz. If we're on the first step, assume wall temperature = freestream temperature.
                    if i == 0:
                        gamma = self.perfect_gas.gamma
                        R = self.perfect_gas.R
                        D = 2*self.y(x)            #Flow diameter

                        #Freestream properties
                        p_inf = self.p(x)
                        T_inf = T_gas[i]
                        rho_inf = self.rho(x)
                        M_inf = self.M(x)
                        v_inf = M_inf * (gamma*R*T_inf)**0.5    #Gas velocity
                        mu_inf = mu_gas[i]
                        Pr_inf = Pr_gas[i]
                        cp_inf = self.perfect_gas.cp

                        ##Properties at arithmetic mean of T_wall and T_inf. Assume wall temperature = freestream temperature for the first step.
                        T_am = T_inf
                        mu_am = self.x_ehaust_transport.mu(T = T_am, p = p_inf)
                        rho_am = p_inf/(R*T_am)                                 #p = rho R T - pressure is roughly uniform across the boundary layer so p_inf ~= p_wall

                        #Stagnation properties
                        p0 = self.chamber_conditions.p0
                        T0 = self.chamber_conditions.T0
                        mu0 = self.x_ehaust_transport.mu(T =  T0, p = p0)

                        h_gas[i] = cool.h_gas_bartz(D, cp_inf, mu_inf, Pr_inf, rho_inf, v_inf, rho_am, mu_am, mu0)
                         
                    else:
                        gamma = self.perfect_gas.gamma
                        R = self.perfect_gas.R
                        D = 2*self.y(x)            #Flow diameter

                        #Freestream properties
                        p_inf = self.p(x)
                        T_inf = T_gas[i]
                        rho_inf = self.rho(x)
                        M_inf = self.M(x)
                        v_inf = M_inf * (gamma*R*T_inf)**0.5    #Gas velocity
                        mu_inf = mu_gas[i]
                        Pr_inf = Pr_gas[i]
                        cp_inf = self.perfect_gas.cp

                        #Properties at arithmetic mean of T_wall and T_inf
                        T_am = (T_inf + T_wall_inner[i-1]) / 2
                        mu_am = self.x_ehaust_transport.mu(T = T_am, p = p_inf)
                        rho_am = p_inf/(R*T_am)                                 #p = rho R T - pressure is roughly uniform across the boundary layer so p_inf ~= p_wall

                        #Stagnation properties
                        p0 = self.chamber_conditions.p0
                        T0 = self.chamber_conditions.T0
                        mu0 = self.x_ehaust_transport.mu(T =  T0, p = p0)

                        h_gas[i] = cool.h_gas_bartz(D, cp_inf, mu_inf, Pr_inf, rho_inf, v_inf, rho_am, mu_am, mu0)

                elif h_gas_model == "bartz-sigma":
                    #We need the previous wall temperature to use h_gas_bartz_sigma. If we're on the first step, assume wall temperature = freestream temperature.
                    if i == 0:
                        h_gas[i] = cool.h_gas_bartz_sigma(self.c_star,
                                                self.nozzle.At, 
                                                self.A(x), 
                                                self.chamber_conditions.p0, 
                                                self.chamber_conditions.T0, 
                                                self.M(x), 
                                                T_gas[i], 
                                                mu_gas[i], 
                                                self.perfect_gas.cp, 
                                                self.perfect_gas.gamma, 
                                                Pr_gas[i])

                    #For all other steps
                    else:
                        h_gas[i] = cool.h_gas_bartz_sigma(self.c_star,
                                                self.nozzle.At, 
                                                self.A(x), 
                                                self.chamber_conditions.p0, 
                                                self.chamber_conditions.T0, 
                                                self.M(x), 
                                                T_wall_inner[i-1], 
                                                mu_gas[i], 
                                                self.perfect_gas.cp, 
                                                self.perfect_gas.gamma, 
                                                Pr_gas[i])

                else:
                    raise AttributeError(f"Could not find the h_gas_model '{h_gas_model}'. Try 'rpe', 'bartz' or 'bartz-sigma'.")

                #Calculate the current coolant temperature
                if i == 0:
                    T_coolant[i] = self.cooling_jacket.inlet_T
                    p0_coolant[i] = self.cooling_jacket.inlet_p0
                    p_coolant[i] = p0_coolant[i] - self.coolant_dynamic_pressure(T=T_coolant[i], p=p0_coolant[i], x=x, y=self.y(x))

                    if p_coolant[i] <= 0:
                        raise ValueError(f"Initial coolant static pressure was negative ({p_coolant[i]} Pa). Try raising the coolant inlet pressure or reducing coolant velocities.")

                    if use_coolant_enthalpy:
                        enthalpy_coolant[i] = self.cooling_jacket.coolant_transport.enthalpy_from_T(T_coolant[i], p_coolant[i])

                else:
                    if use_coolant_enthalpy:
                        #Increase in coolant temperature, q*dx = mdot*dh (if user specified enthalpy function for coolant)
                        dh = (q_dot[i-1]*dx)/self.cooling_jacket.mdot_coolant
                        enthalpy_coolant[i] = enthalpy_coolant[i-1] + dh
                        T_coolant[i] = self.cooling_jacket.coolant_transport.given_T_from_enthalpy(enthalpy_coolant[i], p_coolant[i-1])
                        
                    else:
                        #Increase in coolant temperature, q*dx = mdot*Cp*dT (if user hasn't specified enthalpy function)
                        dT = (q_dot[i-1]*dx)/(self.cooling_jacket.mdot_coolant*cp_coolant[i-1])
                        T_coolant[i] = T_coolant[i-1] + dT

                    #Pressure drop in coolant channel
                    friction_factor = self.coolant_friction_factor(T = T_coolant[i], 
                                                                   p = p_coolant[i-1], 
                                                                   x = x, 
                                                                   y = self.y(x, up_to = "wall out"))

                    p0_coolant[i] = p0_coolant[i-1] - self.coolant_p0_drop(friction_factor, 
                                                                           dl = channel_length[i-1], 
                                                                           T = T_coolant[i], 
                                                                           p = p_coolant[i-1], 
                                                                           x = x, 
                                                                           y = self.y(x, up_to = "wall out"))

                    #Update static pressure of coolant
                    p_coolant[i] = p0_coolant[i] - self.coolant_dynamic_pressure(T = T_coolant[i], 
                                                                                 p = p_coolant[i-1],
                                                                                 x = x, 
                                                                                 y = self.y(x, up_to = "wall out")) 

                    if p_coolant[i] <= 0:
                        raise ValueError(f"Coolant static pressure became negative at x = {x}. Try raising the coolant inlet pressure or reducing coolant velocities.")

                    if too_low_pressure == False and p0_coolant[i] < self.chamber_conditions.p0:
                        too_low_pressure = True
                        print(f"Coolant stagnation pressure dropped below chamber pressure ({self.chamber_conditions.p0/1e5} bar) at x = {x}, the coolant would not flow in real life.")
                    

                #Update coolant heat capacity, transport properties and velocity
                Pr_coolant[i] = self.cooling_jacket.coolant_transport.Pr(T = T_coolant[i], p = p_coolant[i])
                cp_coolant[i] = self.cooling_jacket.coolant_transport.cp(T = T_coolant[i], p = p_coolant[i])
                mu_coolant[i] = self.cooling_jacket.coolant_transport.mu(T = T_coolant[i], p = p_coolant[i])
                k_coolant[i] = self.cooling_jacket.coolant_transport.k(T = T_coolant[i], p = p_coolant[i])
                rho_coolant[i] = self.cooling_jacket.coolant_transport.rho(T = T_coolant[i], p = p_coolant[i])
                v_coolant[i] = self.cooling_jacket.coolant_velocity(rho_coolant[i], x=x, y = self.y(x, up_to = "wall in"))

                #Coolant side heat transfer coefficient
                if h_coolant_model == "rpe":
                    h_coolant[i] = cool.h_coolant_rpe(self.cooling_jacket.A(x=x, y=self.y(x=x, up_to = "wall in")), 
                                                    self.cooling_jacket.D(x=x, y=self.y(x=x, up_to = "wall in")), 
                                                    self.cooling_jacket.mdot_coolant, 
                                                    mu_coolant[i], 
                                                    k_coolant[i], 
                                                    cp_coolant[i], 
                                                    rho_coolant[i])

                elif h_coolant_model == "sieder-tate":
                    #This model requires the cooling channel wall temperature, which hasn't been calculated in the first step. 
                    #Assume the wall temperature = coolant temperature for the first step.
                    if i == 0:
                        h_coolant[i] = cool.h_coolant_sieder_tate(rho_coolant[i], 
                                                                    v_coolant[i], 
                                                                    self.cooling_jacket.D(x=x, y=self.y(x=x, up_to = "wall in")), 
                                                                    mu_coolant[i], 
                                                                    self.cooling_jacket.coolant_transport.mu(T = T_coolant[i], p = p_coolant[i]), 
                                                                    Pr_coolant[i], 
                                                                    k_coolant[i])

                    else:
                        h_coolant[i] = cool.h_coolant_sieder_tate(rho_coolant[i], 
                                                        v_coolant[i], 
                                                        self.cooling_jacket.D(x=x, y=self.y(x=x, up_to = "wall in")), 
                                                        mu_coolant[i], 
                                                        self.cooling_jacket.coolant_transport.mu(T = T_wall_outer[i-1], p = p_coolant[i]), 
                                                        Pr_coolant[i], 
                                                        k_coolant[i])

                elif h_coolant_model == "dittus-boelter":
                    h_coolant[i] = cool.h_coolant_dittus_boelter(rho_coolant[i], 
                                                                v_coolant[i], 
                                                                self.cooling_jacket.D(x=x, y=self.y(x=x, up_to = "wall in")), 
                                                                mu_coolant[i], 
                                                                Pr_coolant[i], 
                                                                k_coolant[i])

                else:
                    raise AttributeError(f"Could not find the h_coolant_model '{h_coolant_model}'. Try 'rpe', 'sieder-tate' or 'dittus-boelter'.")

                #Thermal circuit analysis
                #Combined ablative and regen:
                if self.has_ablative and self.ablative.xs[0] <= x <= self.ablative.xs[1]:
                    #Geometry
                    r_ablative_in = self.y(x, up_to = 'ablative in')
                    r_ablative_out = self.y(x, up_to = 'ablative out')

                    r_wall_in = self.y(x, up_to = 'wall in')
                    r_wall_out = self.y(x, up_to = 'wall out')

                    #Areas per unit length (i.e. just circumference)
                    A_gas = 2 * np.pi * self.y(x, up_to = 'contour')    
                    A_coolant = 2 * np.pi * r_wall_out

                    #Thermal resistances
                    R_gas[i] = gas_fudge_factor*1/(h_gas[i]*A_gas)
                    R_wall[i] = np.log(r_wall_out/r_wall_in)/(2*np.pi*self.ablative.wall_material.k)
                    R_coolant[i] = 1/(h_coolant[i]*A_coolant)
                    R_ablative[i] = np.log(r_ablative_out/r_ablative_in)/(2*np.pi*self.ablative.ablative_material.k)

                    #Thermal circuit object
                    thermal_circuit = cool.ThermalCircuit(T_gas[i], T_coolant[i], [R_gas[i],  R_ablative[i], R_wall[i], R_coolant[i]])

                    q_dot[i] = thermal_circuit.Qdot
                    T_ablative_inner[i] = thermal_circuit.T[1]
                    T_wall_inner[i] = thermal_circuit.T[2]
                    T_wall_outer[i] = thermal_circuit.T[3]

                #Regen but no ablative:
                else:
                    #Geometry
                    r_wall_in = self.y(x, up_to = 'wall in')
                    r_wall_out = self.y(x, up_to = 'wall out')

                    #Areas per unit length (i.e. just circumference)
                    A_gas = 2 * np.pi * self.y(x, up_to = 'contour')    
                    A_coolant = 2 * np.pi * r_wall_out                

                    R_gas[i] = gas_fudge_factor*1/(h_gas[i]*A_gas)
                    R_wall[i] = np.log(r_wall_out/r_wall_in)/(2*np.pi*self.cooling_jacket.inner_wall_material.k)
                    R_coolant[i] = 1/(h_coolant[i]*A_coolant)

                    #Thermal circuit object
                    thermal_circuit = cool.ThermalCircuit(T_gas[i], T_coolant[i], [R_gas[i], R_wall[i], R_coolant[i]])

                    q_dot[i] = thermal_circuit.Qdot
                    T_wall_inner[i] = thermal_circuit.T[1]
                    T_wall_outer[i] = thermal_circuit.T[2]

            else:
                T_wall_inner[i] = T_gas[i]
                T_wall_outer[i] = T_gas[i]

        #Dictionary containing results
        output_dict = {"x" : list(discretised_x),
                "q_dot" : list(q_dot),
                "T_ablative_inner": list(T_ablative_inner),
                "T_wall_inner" : list(T_wall_inner),
                "T_wall_outer" : list(T_wall_outer),
                "T_coolant" : list(T_coolant),
                "T_gas" : list(T_gas),
                "h_gas" : list(h_gas),
                "h_coolant" : list(h_coolant),
                "R_gas" : list(R_gas),
                "R_ablative" : list(R_ablative),
                "R_wall" : list(R_wall),
                "R_coolant" : list(R_coolant),
                "p_coolant" : list(p_coolant),
                "p0_coolant" : list(p0_coolant),
                "mu_gas" : list(mu_gas),
                "k_gas" : list(k_gas),
                "Pr_gas" : list(Pr_gas),
                "Pr_coolant" : list(Pr_coolant),
                "mu_coolant" : list(mu_coolant),
                "k_coolant" : list(k_coolant),
                "cp_coolant" : list(cp_coolant),
                "rho_coolant" : list(rho_coolant),
                "v_coolant" : list(v_coolant),
                "enthalpy_coolant" : list(enthalpy_coolant)}

        #Export a .JSON file if required
        if to_json != False:
            with open(to_json, "w+") as write_file:
                json.dump(output_dict, write_file)
                print("Exported JSON data to '{}'".format(to_json))

        return output_dict

    def transient_heating_analysis(self, number_of_points=1000, dt = 0.1, t_max = 100, wall_starting_T = 298.15, h_gas_model = "1", to_json = "heating_output.json"):
        """This is used exclusive for pure ablative cooling, without any regenerative cooling jacket.

        Note:
            This function is outdated and likely no longer functional, as it does not incorporate many new features that have been added to Bamboo.

        Args:
            number_of_points (int, optional): [description]. Defaults to 1000.
            dt (float, optional): Timestep (s). Defaults to 0.1.
            t_max (float, optional): Maximum time to run to (s). Defaults to 100
            wall_starting_T (float, optional): Starting temperature for the wall (K). Defaults to 298.15.
            h_gas_model (str, optional): [description]. Defaults to "1".
            to_json (str, optional): [description]. Defaults to "heating_output.json".


        """
        try:
            self.geometry
        except AttributeError:
            raise AttributeError("Cannot run heating analysis without additional geometry definitions. You need to add geometry with the 'Engine.add_geometry()' function.")

        try:
            self.x_ehaust_transport
        except AttributeError:
            raise AttributeError("Cannot run heating analysis without an exhaust gas transport properties model. You need to add one with the 'Engine.add_exhaust_transport()' function.")

        try:
            self.ablative
        except AttributeError:
            raise AttributeError("Cannot run heating analysis without an exhaust gas transport properties model. You need to add one with the 'Engine.add_exhaust_transport()' function.")

        print("Starting transient heating analysis")

        '''Initialise variables and arrays'''
        #Discretisation of the nozzle
        discretised_x = np.linspace(self.geometry.x_max, self.geometry.x_min, number_of_points) #Run from the back end (the nozzle exit) to the front (chamber)
        discretised_t = np.arange(0, t_max, dt)
        dx = discretised_x[0] - discretised_x[1]

        #Discretised liner thickness
        liner = self.map_thickness_profile(self.geometry.inner_wall_thickness, number_of_points)

        #Temperatures and heat transfer rates - everything is a 2D array, with indexes [time_index, space_index]
        T_wall = np.zeros([len(discretised_t), len(discretised_x)]) 
        T_ablative_inner = np.zeros([len(discretised_t), len(discretised_x)]) 
        T_ablative_outer = np.zeros([len(discretised_t), len(discretised_x)]) 
        T_gas = np.zeros([len(discretised_t), len(discretised_x)]) 
        q_dot = np.zeros([len(discretised_t), len(discretised_x)]) 
        h_gas = np.zeros([len(discretised_t), len(discretised_x)]) 

        '''Main loop'''
        for i in range(len(discretised_t)):
            if i%50 == 0:
                print(f"{100*i/len(discretised_t)}% complete")

            for j in range(len(discretised_x)):
                t = discretised_t[i]
                x = discretised_x[j]

                #Freestream gas temperature 
                T_gas[i, j] = self.T(x)

                #Ablative thickness (currently a placeholder for custom thicknesses)
                ablative_thickness = self.geometry.chamber_radius - self.y(x)

                #Gas side heat transfer coefficient
                if h_gas_model == "1":
                    h_gas[i, j] = cool.h_gas_1(2*self.y(x),
                                            self.M(x),
                                            T_gas[i, j],
                                            self.rho(x),
                                            self.perfect_gas.gamma,
                                            self.perfect_gas.R,
                                            self.x_ehaust_transport.mu(T = T_gas[i, j], p = self.p(x)),
                                            self.x_ehaust_transport.k(T = T_gas[i, j], p = self.p(x)),
                                            self.x_ehaust_transport.Pr(T = T_gas[i, j], p = self.p(x)))

                elif h_gas_model == "2":
                    gamma = self.perfect_gas.gamma
                    R = self.perfect_gas.R
                    D = 2*self.y(x)            #Flow diameter

                    #Freestream properties
                    p_inf = self.p(x)
                    T_inf = T_gas[i, j]
                    rho_inf = self.rho(x)
                    M_inf = self.M(x)
                    v_inf = M_inf * (gamma*R*T_inf)**0.5    #Gas velocity
                    mu_inf = self.x_ehaust_transport.mu(T = T_gas[i, j], p = p_inf)
                    Pr_inf = self.x_ehaust_transport.Pr(T = T_gas[i, j], p = p_inf)
                    cp_inf = self.perfect_gas.cp

                    #Properties at arithmetic mean of T_wall and T_inf
                    T_am = (T_inf + T_ablative_inner[i, j-1]) / 2
                    mu_am = self.x_ehaust_transport.mu(T = T_am, p = p_inf)
                    rho_am = p_inf/(R*T_am)                                 #p = rho R T - pressure is roughly uniform across the boundary layer so p_inf ~= p_wall

                    #Stagnation properties
                    p0 = self.chamber_conditions.p0
                    T0 = self.chamber_conditions.T0
                    mu0 = self.x_ehaust_transport.mu(T =  T0, p = p0)

                    h_gas[i, j] = cool.h_gas_2(D, cp_inf, mu_inf, Pr_inf, rho_inf, v_inf, rho_am, mu_am, mu0)

                elif h_gas_model == "3":
                    h_gas[i, j] = cool.h_gas_3(self.c_star,
                                            self.nozzle.At, 
                                            self.A(x), 
                                            self.chamber_conditions.p0, 
                                            self.chamber_conditions.T0, 
                                            self.M(x), 
                                            T_ablative_inner[i, j-1], 
                                            self.x_ehaust_transport.mu(T = T_gas[i, j], p = self.p(x)), 
                                            self.perfect_gas.cp, 
                                            self.perfect_gas.gamma, 
                                            self.x_ehaust_transport.Pr(T = T_gas[i, j], p = self.p(x)))

                else:
                    raise AttributeError(f"Could not find the h_gas_model '{h_gas_model}'")

                #Get thermal circuit properties
                q_dot[i, j], R_gas, R_ablative = self.ablative_thermal_circuit(self.y(x), 
                                                                                   h_gas[i, j], 
                                                                                   self.ablative.ablative_material, 
                                                                                   ablative_thickness, 
                                                                                   T_gas[i, j], 
                                                                                   T_wall[i-1, j])

                #Calculate wall temperatures
                T_ablative_inner[i, j] = T_gas[i, j] - q_dot[i, j]*R_gas
                T_ablative_outer[i, j] = T_ablative_inner[i, j] - q_dot[i, j]*R_ablative

                #Find the rise in wall temperature - assume only radial heat transfer
                if i == 0:
                    T_wall[i, j] = wall_starting_T

                else:
                    dV = np.pi*((self.y(x) + ablative_thickness + self.ablative.wall_thickness)**2 - (self.y(x) + ablative_thickness)**2)*dx         #Volume of wall
                    dm = dV*self.ablative.wall_material.rho                                    #Mass of wall

                    #q_dot*dx*dt = m*c*dT
                    T_wall[i, j] = T_wall[i-1, j] + q_dot[i, j]*dx*dt/(self.ablative.wall_material.c*dm)

        '''Return results'''
        #Dictionary containing results
        output_dict = {"x" : list(discretised_x),
                        "t" : list(discretised_t),
                        "T_wall" : list(T_wall),
                        "T_ablative_inner" : list(T_ablative_inner),
                        "T_ablative_outer" : list(T_ablative_outer),
                        "T_gas" : list(T_gas),
                        "q_dot" : list(q_dot),
                        "h_gas" : list(h_gas)}

        #Export a .JSON file if required
        if to_json != False:
            with open(to_json, "w+") as write_file:
                json.dump(output_dict, write_file)
                print("Exported JSON data to '{}'".format(to_json))

        return output_dict


    #Stress analysis functions
    def run_stress_analysis(self, heating_result, condition="steady", **kwargs):
        """Perform stress analysis on the liner, using a cooling result.
           Results should be taken only as a first approximation of some key stresses.

        Note:
            Will likely not work right now, as it has not been updated to accodomate variable cooling channel heights

        Args:
            heating_result (dict): Requires a heating analysis result to compute stress.
            condition (str, optional): Engine state for analysis. Options are "steady", or "transient". Defaults to "steady".

        Keyword Args:
            T_amb (float, optional): For transient analysis, the ambient temperature can be overriden from the default, 283 K.
                                     This is used for calculating thermal expansions, so it's really the zero thermal stress
                                     temperature, presumably the temperature at which the engine was assembled.

        Returns:
            dict: Results of the stress simulation. Contains the following dictionary keys (all are arrays): 
                - "thermal_stress : (steady only) Stress induced in the inner liner due to temperature difference, chamber to coolant side, for each x value (Pa).
                - "tadjusted_yield : (steady only) Yield stress of the inner wall material, corrected for the chamber side temperature (worst case) (Pa).
                - "deltaT_wall" : (steady only) Inner liner temperature difference, chamber side - coolant side (K).
                - "stress_inner_hoop_steady" : (steady only) Hoop stress of inner liner due to coolant static pressure in jacket after ignition (Pa).
                - "stress_outer_hoop" : (steady only) Hoop stress of outer liner due to coolant static pressure (same before and after ignition) (Pa).
                - "stress_inner_hoop_transient" : (transient only) Hoop stress of inner liner due to coolant static pressure in jacket prior to ignition (0 chamber pressure) (Pa).
                - "stress_inner_IE" : (transient only) Stress induced in inner liner as it is heated but constrained by cold outer liner (Pa).
                - "stress_outer_IE : (transient only) Stress induced in outer liner by expanding inner liner (Pa).
        """
        length = len(heating_result["x"])
        wall_stress = np.zeros(length)
        wall_deltaT = np.zeros(length)
        tadjusted_yield = np.zeros(length)
        mat_in = self.cooling_jacket.inner_wall_material
        mat_out = self.cooling_jacket.outer_wall_material

        print("WARNING: The stress analysis function has not been updated to accommodate variable channel heights, and so may not work correctly at the moment")

        E1 = self.cooling_jacket.inner_wall_material.E
        E2 = self.cooling_jacket.outer_wall_material.E
        t1_hoop = self.map_thickness_profile(self.geometry.inner_wall_thickness, length)
        t1 = t1_hoop + self.cooling_jacket.channel_height(x)*self.cooling_jacket.blockage_ratio         #WARNING: Probably need to fix this line to accommodate variable channel heights
        # The blockage ratio is used to scale the contribution of the ribs to the
        # effective total thickness of the inner liner (wider ribs = greater blockage ratio)
        t2 = self.map_thickness_profile(self.geometry.outer_wall_thickness, length)

        # Geometry calculations used for non-thermal stresses;
        # if there is an ablative, the inner liner radius will be
        # constant as the nozzle geometry is created with this instead of
        # the cooling jacket.
        if self.has_ablative is True or self.cooling_jacket.has_ablative is True:
            if self.ablative.wall_material != self.cooling_jacket.inner_wall_material:
                raise AttributeError("Change of material behind ablator is not "
                                     "currently supported in stress analysis")
            R1 = [self.geometry.chamber_radius]*length + t1/2
            R2 = R1 + t1/2 + t2/2
            R1_hoop = [self.geometry.chamber_radius]*length
        else:
            R1 = self.geometry.y(x, up_to="wall in") + t1/2
            R2 = R1 + t1/2 + t2/2
            R1_hoop = self.geometry.y(x, up_to="wall in")
        
        if condition == "steady":
            for i in range(length):
                wall_deltaT[i] = heating_result["T_wall_inner"][i] - \
                    heating_result["T_wall_outer"][i]
                tadjusted_yield[i] = mat_in.relStrength(heating_result["T_wall_inner"][i]) * mat_in.sigma_y

            for i in range(length):
                cur_stress = mat_in.k * \
                    wall_deltaT[i] / (2 * mat_in.perf_therm)
            # Determine thermal stress using Ref [7], P53:
            # sigma_thermal = E*alpha*q_w*deltaL/(2*(1-v)k_w) =
            # E*alpha*deltaT/2(1-v)

                wall_stress[i] = cur_stress

            # Now we determine the hoop stresses. For the outer liner,
            # the pressure is taken to be the coolant pressure.
            # In reality the average pressure is less than this, because
            # the ribs occupy part of the wall. 
            # Ambient pressure is also neglected for the outer liner.

            sigma_inner_hoop = np.array(heating_result["p_coolant"]) * R1_hoop/t1_hoop
            sigma_outer_hoop = np.array(heating_result["p_coolant"])*R2/t2

            return {"thermal_stress": wall_stress,
                    "yield_adj": tadjusted_yield,
                    "deltaT_wall": wall_deltaT,
                    "stress_inner_hoop_steady": sigma_inner_hoop,
                    "stress_outer_hoop": sigma_outer_hoop}

        if condition == "transient":
            if "T_amb" in kwargs:
                T_amb = kwargs["T_amb"]
            else:
                T_amb = 283
            # T_amb is really the zero thermal stress temperature for the jacket, which
            # I think should be the temperature at which the engine was assembled?

            if self.cooling_jacket.configuration == "vertical":
                # Assumptions for this analysis:
                # Both liners must not be axially constrained, i.e. free at
                # at least one end. This may present a challenge for sealing the
                # outer liner, but as the outer liner temperature changes are realtively
                # low this should not problematic.

                # The ribs are treated as a ring exerting uniform pressure around the
                # circumference on the outer liner, which is not strictly true.

                # Finally, the chamber side temperature of the inner wall is used for
                # determining the inner liner expansion, which is actually governed by the
                # temperature of the top surface of the ribs, which contact the outer liner.

                epsilon_T = np.zeros(length)

                # Use chamber radius if there is an ablator - the inner liner
                # radius will be constant as the ablative handles the nozzle contour

                epsilon_T = (np.array((heating_result["T_wall_inner"])) - T_amb) \
                                * mat_in.alpha
                # Find the (unconstrained) thermal strain where the ribs
                # meet the outer liner, for each x position

                # Now impose this expansion on the outer liner to find the equlibrium.
                # By compatibility, both liners must have the same change in radius
                # By taking a cut of half of the section and imposing equlibrium:
                # sigma_inner = -alpha*deltaT*R1 / ((R1/E1) + (R2*t1)/(E2*t2))
                # sigma_outer = -t1*sigma_inner/t2
                # R1 = inner liner average radius, R2 = outer liner average radius
                # t1 = wall height + rib height, t2 = outer liner thickness
                # E1 = inner liner Young's modulus, E2 = outer liner Young's modulus

                sigma_inner_IE = -epsilon_T * R1/((R1/E1) + ((R2*t1)/(E2*t2)))
                sigma_outer_IE = -sigma_inner_IE*t1/t2
                # _IE = due to inner expansion

                # Now determine the startup inner liner hoop stress
                # Array for the pressure inside the engine along its length, after ignition.
                discretised_x = np.linspace(self.geometry.x_max, self.geometry.x_min, length)
                engine_pressure = [self.p(discretised_x[i]) for i in range(length)]
                sigma_inner_hoop = (np.array(heating_result["p_coolant"]) - \
                                    engine_pressure) * R1_hoop/t1_hoop

                return {"stress_inner_hoop_transient": sigma_inner_hoop,
                        "stress_inner_IE": sigma_inner_IE,
                        "stress_outer_IE": sigma_outer_IE}

            else:
                raise AttributeError("Currently only vertical channels are supported for"
                                     " transient stress analysis.")
