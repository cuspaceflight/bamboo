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
        print("bamboo.main.rao_theta_n(): Area ratio is outside the range of the Rao inflection angle data, returning 15 deg instead.")
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
        print("bamboo.main.rao_theta_e(): Area ratio is outside the range of the Rao exit angle data, returning 14.999 deg instead.")
        return 14.999*np.pi/180
        #raise ValueError(f"The area ratio provided ({area_ratio}) is outside of the range of available data. Maximum available is {data['area_ratio'][-1]}, minimum is {data['area_ratio'][0]}.")
    
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
        At (float): Throat area (m^2).
        Ae (float): Exit plane area (m^2)
        type (str, optional): Desired shape, can be "rao" or "conical". Conical is not currently implemented. Defaults to "rao".
        length_fraction (float): Length fraction if a Rao nozzle is used. Defaults to 0.8.
        cone_angle (float): Cone angle if a cone nozzle is used (deg). Defaults to 15.
    
    Attributes:
        length (float): Length of the diverging section (distance between throat and nozzle exit) (m).
        At (float): Throat area (m^2)
        Ae (float): Exit area (m^2)
        Rt (float): Throat radius (m)
        Re (float): Exit radius (m)
    """
    def __init__(self, At, Ae, type = "rao", length_fraction = 0.8, cone_angle = 15):
        self.At = At
        self.Ae = Ae
        self.Rt = (At/np.pi)**0.5   #Throat radius (m)
        self.Re = (Ae/np.pi)**0.5   #Exit radius (m)
        self.type = type
        
        if self.type == "cone":
            self.cone_angle = cone_angle
            self.dydx = np.tan(self.cone_angle*np.pi/180)
            self.length = (self.Re - self.Rt)/self.dydx

        elif self.type == "rao":
            self.length_fraction = length_fraction
            self.theta_n = rao_theta_n(self.Ae/self.At)     #Inflection angle (rad), as defined in [1]
            self.theta_e = rao_theta_e(self.Ae/self.At)     #Exit angle (rad), as defined in [1]

            #Page 5 of Reference [1]:
            self.Nx = 0.382*self.Rt*np.cos(self.theta_n - np.pi/2)                              #Inflection point x-value
            self.Ny = 0.382*self.Rt*np.sin(self.theta_n - np.pi/2) + 0.382*self.Rt + self.Rt    #Inflection point y-value
            self.Ex = 0.8*((self.Re/self.Rt) - 1)*self.Rt/np.tan(np.pi/12)                      #Exit x-value
            self.Ey = self.Re                                                                   #Exit y-value (same as Re)
            self.length = self.Ex                                                               #Nozzle length

            #Similar to page 2 of Reference [3]. Set up the matrix problem to get the coefficients for x = ay^2 + by + c
            #We will fit the quadratic using the inflection point and exit coordinates, and the exit gradient. The inflection gradient is ignored.
            A = np.array([[2*self.Ey, 1, 0],
                          [self.Ny**2, self.Ny, 1],
                          [self.Ey**2, self.Ey, 1]], dtype='float')
            
            b = np.array([1/np.tan(self.theta_e), self.Nx, self.Ex], dtype='float')

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
    """Container for additional engine geometry parameters (mostly chamber geometry). Used internally for heating analyses.

    Using the 'inner_wall_thickness' (or 'outer_wall_thickness', if also provided) argument:
        If an array, must be thicknesses at equally spaced x positions. This will be stretched to fill the engine length.
        E.g. [1e-3, 5e-3] will have 1mm thick walls at chamber entrance, 5mm thick at nozzle exit.

    Args:
        nozzle (float): Nozzle of the engine.
        chamber_length (float): Length of the combustion chamber (m)
        chamber_area (float): Cross sectional area of the combustion chamber (m^2)
        inner_wall_thickness (float or array): Thickness of the inner liner wall (m). Can be constant (float), or variable (array). 
        outer_wall_thickness (float or array): Thickness of the outer liner wall (m). Can be constant (float), or variable (array).
        style (str, optional): Geometry style to use. Currently the only option is 'auto'. Defaults to "auto".

    Attributes:
        x_min (float): Minimum x position (m).
        x_max (float): Maximum x position (m).
        x_chamber_end (float): x position where the combustion chamber ends (m).
        x_curved_converging_start (float): x position where the curved part of the converging section begins (m).
        chamber_length (float): Chamber length (m).
        chamber_area (float): Chamber area (m^2).
        chamber_radius (float): Chamber radius (m).
        

    """
    def __init__(self, nozzle, chamber_length, chamber_area, inner_wall_thickness, outer_wall_thickness, style="auto"):
        self.chamber_length = chamber_length
        self.chamber_area = chamber_area
        self.chamber_radius = (chamber_area/np.pi)**0.5 
        if type(inner_wall_thickness) is float or type(inner_wall_thickness) is int:
            self.inner_wall_thickness = [inner_wall_thickness]  #Convert into a list so the interpolation works
        else:
            self.inner_wall_thickness = inner_wall_thickness
        
        if type(outer_wall_thickness) is float or type(outer_wall_thickness) is int:
            self.outer_wall_thickness = [outer_wall_thickness]  #Convert into a list so the interpolation works
        else:
            self.outer_wall_thickness = outer_wall_thickness

        if nozzle.At > self.chamber_area:
            raise ValueError(f"The combustion chamber area {self.chamber_area} m^2 is smaller than the throat area {nozzle.At} m^2.")
        
        self.style = style
        if self.style == "auto":
            if nozzle.type == "cone":
                self.dydx_conv = np.tan(-45*np.pi/180)
                self.x_max = nozzle.length
                self.x_chamber_end = (self.chamber_radius - nozzle.Rt)/self.dydx_conv
                self.x_min = self.x_chamber_end - self.chamber_length

            elif nozzle.type == "rao":
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
                self.x_max = nozzle.length

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
        max_throat_area = get_throat_area(perfect_gas, chamber_conditions)
        if self.nozzle.At > max_throat_area:
            raise ValueError(f"The nozzle throat is not choked. You need to reduce the throat area to at least {max_throat_area} m^2")

    #Engine geometry functions
    def y(self, x, up_to = 'contour'):
        """Get y position up to a specified part of the engine (e.g. inner contour, ablative inner or outer wall, etc.)

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

            #Converging section and combustion chamber
            else:
                try:
                    self.geometry
                except AttributeError:
                    raise AttributeError("Geometry is not defined for x < 0. You need to add geometry with the 'Engine.add_geometry()' function.")

                if self.geometry.style == "auto":
                    if self.nozzle.type == "rao":
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

                            #Outside of the engine
                            else:
                                raise ValueError(f"x is beyond the front of the engine. You tried to input {x} but the minimum value you're allowed is {self.geometry.x_chamber_end - self.geometry.chamber_length}")
                    
                    elif self.nozzle.type == "cone":
                        #If between the end of the chamber and the throat
                        if x < 0 and x >= self.geometry.x_chamber_end:
                            #Use a 45 degree converging section
                            return self.nozzle.Rt + self.geometry.dydx_conv*x

                        #If in the chamber
                        elif x >= self.geometry.x_min and x < self.geometry.x_chamber_end:
                            return self.geometry.chamber_radius

                        else:
                            raise ValueError(f"x is beyond the front of the engine. You tried to input {x} but the minimum value you're allowed is {self.geometry.x_min}")
                    

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
            if self.ablative.xs[0] < x < self.ablative.xs[1]:
                if self.ablative.ablative_thickness == None:
                    #If self.ablative_thickness == None, fill up the distance between the nozzle contour and the chamber radius with ablative
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
                Mach = scipy.optimize.root_scalar(func_to_solve, bracket = [1,300], x0 = 1).root
            else:
                Mach = scipy.optimize.root_scalar(func_to_solve, bracket = [0,1], x0 = 0.5).root

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

    
    #Thrust and performance functions
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
                    D = self.cooling_jacket.D(x[0], wall_outer[0])
                    A = self.cooling_jacket.A(x[0], wall_outer[0])

                    regen_xs = np.linspace(xmin, xmax, int((xmax - xmin)/D))

                    if len(regen_xs) > 5000:
                        print(f"WARNING: Large number of channels to plot for the cooling jacket ({len(regen_xs)}) - this may take a while.")

                    axs.plot(0, 0, color = 'green', label = 'Cooling channels')  #Just for the legend

                    for i in range(len(regen_xs) - 1):
                        y_jacket_inner = np.interp(regen_xs[i], x, wall_outer)

                        #We'll show the coolant channels as rectangles, with a diameter equal to the equivelant diameter, and area equal to the flow area.
                        axs.add_patch(matplotlib.patches.Rectangle([regen_xs[i], y_jacket_inner], D, A/D, color = 'green', fill = False))
                        axs.add_patch(matplotlib.patches.Rectangle([regen_xs[i], -y_jacket_inner-A/D], D, A/D, color = 'green', fill = False))

                #If using a vertical channels cooling jacket
                if self.cooling_jacket.configuration == 'vertical':
                    regen_xs = np.linspace(xmin, xmax, 1000)
                    channel_inner_mapped = np.interp(regen_xs, x, wall_outer)

                    #Show the channel thickness to scale
                    axs.fill_between(regen_xs, channel_inner_mapped, channel_inner_mapped+self.cooling_jacket.channel_height, color="green", label = 'Cooling channel')
                    axs.fill_between(regen_xs, -channel_inner_mapped, -channel_inner_mapped-self.cooling_jacket.channel_height, color="green")
            
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
    def add_geometry(self, chamber_length, chamber_area, inner_wall_thickness, outer_wall_thickness, style="auto"):
        """Specify extra geometry parameters. Required for running cooling system analyses.

        Args:
            nozzle (Nozzle): Nozzle of the engine.
            chamber_length (float): Length of the combustion chamber (m)
            chamber_area (float): Cross sectional area of the combustion chamber (m^2)
            inner_wall_thickness (float or array): Thickness of the inner liner wall (m). Can be constant (float), or variable (array).
            style (str, optional): Geometry style to use. Currently the only option is 'auto'. Defaults to "auto".      
            outer_wall_thickness (float or array): Thickness of the outer liner wall (m). Can be constant (float), or variable (array).
        """


        self.geometry = EngineGeometry(self.nozzle, chamber_length, chamber_area, inner_wall_thickness,
                                       outer_wall_thickness, style)
        self.has_geometry = True

    def add_cooling_jacket(self, inner_wall, outer_wall, inlet_T, inlet_p0, coolant_transport, mdot_coolant, xs = [-1000, 1000], configuration = "spiral", **kwargs):
        """Container for cooling jacket information - e.g. for regenerative cooling.

        Args:
            inner_wall (Material): Inner wall material.
            outer_wall (Material): Wall material for the outer liner. 
            inlet_T (float): Inlet coolant temperature (K)
            inlet_p0 (float): Inlet coolant stagnation pressure (Pa)
            coolant_transport (TransportProperties): Container for the coolant transport properties.
            mdot_coolant (float): Coolant mass flow rate (kg/s)
            xs (list): x position that the cooling jacket starts and ends at in the form [x_start, x_end]. Defaults to [-1000, 1000].
            configuration (str, optional): Options include 'spiral' and 'vertical'. Defaults to "vertical".
        
        Keyword Args:
            channel_shape (str, optional): Used if configuration = 'spiral'. Options include 'rectangle', 'semi-circle', and 'custom'.
            channel_height (float, optional): If using configuration = 'vertical' or channel_shape = 'rectangle', this is the height of the channels (m).
            channel_width (float, optional): If using channel_shape = 'rectangle', this is the width of the channels (m). If using channel_shape = 'semi-circle', this is the diameter of the semi circle (m).
            custom_effective_diameter (float, optional): If using channel_shape = 'custom', this is the effective diameter you want to use.
            custom_flow_area (float, optional): If using channel_shape = 'custom', this is the flow you want to use. 
        """
        
        self.cooling_jacket = cool.CoolingJacket(self.geometry,
                                                inner_wall,
                                                outer_wall, 
                                                inlet_T, 
                                                inlet_p0, 
                                                coolant_transport, 
                                                mdot_coolant, 
                                                xs, 
                                                configuration, 
                                                self.has_ablative,
                                                **kwargs)
        self.has_cooling_jacket = True

    def add_exhaust_transport(self, transport_properties):
        """Add a model for the exhaust gas transport properties (e.g. viscosity, thermal doncutivity, etc.). This is needed to run cooling system analyses.

        Args:
            transport_properties (TransportProperties): Container for the exhaust gas transport properties.
        """
        self.exhaust_transport = transport_properties
        self.has_exhaust_transport = True

    def add_ablative(self, ablative_material, wall_material = None, xs = [-1000, 1000], ablative_thickness = None, regression_rate = 0.0):
        """
        Args:
            ablative_material (Material): Ablative material.
            wall_material (Material): Wall material on the outside of the ablative (will override the cooling jacket wall material). Defaults to None, in which case the cooling jacket material will be used.
            xs (list, optional): x positions that the ablative is present between, [xmin, xmax]. Defaults to [-1000, 1000].
            ablative_thickness (float or list): Thickness of ablative. If a list is given, it must correspond to thickness at regular x intervals, which will be stretched out over the inverval of 'xs'. Defaults to None (in which case the ablative extends from the engine contour to combustion chamber radius).
            regression_rate (float): (Not currently used) (m/s). Defaults to 0.0.
        """
        #Use the cooling jacket's wall material if the user inputs 'wall_material = None'
        
        if wall_material == None:
            if self.has_cooling_jacket == False:
                raise AttributeError("You need to specify a wall material for the ablative (there is no cooling jacket wall material to use)")
            wall_material_to_use = self.cooling_jacket.inner_wall

        else:
            wall_material_to_use = wall_material
        
        if self.has_cooling_jacket is True:
            self.cooling_jacket.has_ablative = True
        # Update the cooling jacket if an ablator is added after the jacket

        self.ablative = cool.Ablative(ablative_material = ablative_material,
                                      wall_material = wall_material_to_use, 
                                      regression_rate = regression_rate, 
                                      xs = xs, 
                                      ablative_thickness = ablative_thickness)
        self.has_ablative = True

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

    def channel_geometry(self, number_of_sections = 1000):
        """Finds the path length of the coolant in the jacket from engine geometry and channel configuration.
           Number_of_sections must be equal to number_of_points when used in a heating analysis.

        Note:
            Recently changed the system for finding the engine contour - would be useful to run some proper tests on this function to see if the values it returns still make sense.

        Args:
            number_of_sections (int, optional): Number of sections to split path into. Defaults to 1000.


        Returns:
            array: Discretised coolant path length array with "number_of_sections" elements. (m).
        """
        discretised_x = np.linspace(self.geometry.x_max, self.geometry.x_min, number_of_sections)
        axis_length = self.geometry.x_max - self.geometry.x_min     # Axial engine length
        discretised_length = []

        if self.cooling_jacket.configuration == "spiral":
            pitch = self.cooling_jacket.channel_width               # No gaps between channels so spiral pitch = width
            section_turns = axis_length/(pitch*number_of_sections)  # Number of turns per discrete section

            for i in range(number_of_sections-1):              
                if self.has_ablative is True:
                    y = self.geometry.chamber_radius
                else:
                    y = self.y(discretised_x[i], up_to = "wall out")
                # Ignore the nozzle contours - jacket has constant radius if an ablative insert is present
                 
                radius_avg = (y + self.y(discretised_x[i+1], up_to = 'wall out'))/2
                discretised_length.append(section_turns * np.sqrt(pitch**2 + (radius_avg*2*np.pi)**2))
                # Find the average radius for this section and use it to determine the spiral section length

            return discretised_length

        if self.cooling_jacket.configuration == "vertical":
            dx = discretised_x[0] - discretised_x[1]

            for i in range(number_of_sections-1):
                dy = np.abs(self.y(discretised_x[i], up_to = 'wall out') - self.y(discretised_x[i+1], up_to = 'wall out'))
                discretised_length.append(np.sqrt(dy**2 + dx**2))

            return discretised_length

        else:
            raise AttributeError("Invalid cooling channel configuration")

    def coolant_friction_factor(self, T, p, x, y = None):
        """Determine the friction factor of the coolant at the current position.
           Formula from reference [5] page 29.
        Args:
            x (float): Axial position
            y (float, optional): The radius of the engine (m) (NOT the radius of the cooling channel).  Only required for 'vertical' channels. 
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

    def Q_coolant(self, T, p, x = None, y = None):
        """Determine dynamic pressure of coolant.

        Args:
            x (float, optional): Axial position. Only required for 'vertical' channels.
            y (float, optional): The radius of the engine (m) (NOT the radius of the cooling channel). Only required for 'vertical' channels.
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
            y (float, optional): The radius of the engine (m) (NOT the radius of the cooling channel). Only required for 'vertical' channels.
            dl (float): Length to evaluate pressure drop over - an increment along the channel, not the engine axis
            T (float): Coolant temperature            
            p (float): Coolant pressure
        Returns:
            float: Stagnation pressure drop (Pa)
        """

        D = self.cooling_jacket.D(x, y)
        Q = self.Q_coolant(T=T, p=p, x=x, y=y)

        return friction_factor*dl*Q/D

    def regen_thermal_circuit(self, r, h_gas, h_coolant, wall_material, inner_wall_thickness, T_gas, T_coolant):
        """
        q is per unit length along the nozzle wall (axially) - positive when heat is flowing to the coolant.   
        Uses the idea of thermal circuits and resistances - we have three resistors in series.

        Args:
            r (float): Radius to the inner wall of the engine (m)
            h_gas (float): Gas side convective heat transfer coefficient
            h_coolant (float): Coolant side convective heat transfer coefficient
            wall_material (Material): Material object for the inner wall, needed for thermal conductivity
            inner_wall_thickness (float): Thickness of the inner wall at x position (m)
            T_gas (float): Free stream gas temperature (K)
            T_coolant (float): Coolant temperature (K)

        Returns:
            float, float, float, float: q_dot, R_gas, R_wall, R_coolant
        """

        r_in = r   
        r_out = r_in + inner_wall_thickness

        A_in = 2*np.pi*r_in         #Inner area per unit length (i.e. just the inner circumference)
        A_out = 2*np.pi*r_out       #Outer area per unit length (i.e. just the outer circumference)

        R_gas = 1/(h_gas*A_in)
        R_wall = np.log(r_out/r_in)/(2*np.pi*wall_material.k)
        R_coolant = 1/(h_coolant*A_out)

        q_dot = (T_gas - T_coolant)/(R_gas + R_wall + R_coolant)    #Heat flux per unit length

        return q_dot, R_gas, R_wall, R_coolant

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

    def regen_ablative_thermal_circuit(self, r, h_gas, h_coolant, wall_material, inner_wall_thickness, T_gas, T_coolant, ablative_material, ablative_thickness):
        """Combined regenerative and ablative cooling thermal circuit.
        q is per unit length along the nozzle wall (axially) - positive when heat is flowing to the coolant.  
        q_Adot is the heat flux per unit area along the nozzle wall.  
        Uses the idea of thermal circuits and resistances - we have three resistors in series.

        Args:
            r (float): Radius to the contour of the engine (m)
            h_gas (float): Gas side convective heat transfer coefficient
            h_coolant (float): Coolant side convective heat transfer coefficient
            wall_material (Material): Material object for the inner wall, needed for thermal conductivity
            inner_wall_thickness (float): Thickness of the inner wall at x position (m)
            T_gas (float): Free stream gas temperature (K)
            T_coolant (float): Coolant temperature (K)
            ablative_material (Material): Material object for the ablative material, needed for thermal conductivity
            ablative_thickness (float): Thickness of the ablative material (m)

        Returns:
            float, float, float, float, float: q_dot, R_gas, R_ablative, R_wall, R_coolant
        """
        
        r_ablative_in = r 
        r_ablative_out = r_ablative_in + ablative_thickness

        r_wall_in = r_ablative_out
        r_wall_out = r_wall_in + inner_wall_thickness

        A_wall_in = 2*np.pi*r_wall_in       #Inner area per unit length (i.e. just the inner circumference)
        A_wall_out = 2*np.pi*r_wall_out     #Outer area per unit length (i.e. just the outer circumference)

        R_gas = 1/(h_gas*A_wall_in)
        R_wall = np.log(r_wall_out/r_wall_in)/(2*np.pi*wall_material.k)
        R_coolant = 1/(h_coolant*A_wall_out)
        R_ablative = np.log(r_ablative_out/r_ablative_in)/(2*np.pi*ablative_material.k)

        q_dot = (T_gas - T_coolant)/(R_gas + R_ablative + R_wall + R_coolant)    #Heat flux per unit length

        return q_dot, R_gas, R_ablative, R_wall, R_coolant,

    def steady_heating_analysis(self, number_of_points=1000, h_gas_model = "1", h_coolant_model = "1", to_json = "heating_output.json"):
        """Steady state heating analysis. Can be used for regenarative cooling, or combined regenerative and ablative cooling.

        Args:
            number_of_points (int, optional): Number of discrete points to divide the engine into. Defaults to 1000.
            h_gas_model (str, optional): Equation to use for the gas side convective heat transfer coefficients. Options are '1', '2' and '3'. Defaults to "1".
            h_coolant_model (str, optional): Equation to use for the coolant side convective heat transfer coefficients. Options are '1', '2' and '3'. Defaults to "1".
            to_json (str or bool, optional): Directory to export a .JSON file to, containing simulation results. If False, no .JSON file is saved. Defaults to 'heating_output.json'.

        Note:
            h_gas_model = '2' seems to provide questionable results (if it works at all) - use it with caution. h_coolant_model = '2' can raise errors if using the 'force_phase' setting with your coolant TransportProperties object. See the functions h_gas_1(), h_gas_2(), h_coolant_1(), etc.. in the documentation for details on each model.

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
                - (and some more)
        """

        '''Check if we have everything needed to run the simulation'''
        try:
            self.geometry
        except AttributeError:
            raise AttributeError("Cannot run heating analysis without additional geometry definitions. You need to add geometry with the 'Engine.add_geometry()' function.")

        try:
            self.exhaust_transport
        except AttributeError:
            raise AttributeError("Cannot run heating analysis without an exhaust gas transport properties model. You need to add one with the 'Engine.add_exhaust_transport()' function.")
        
        if h_gas_model == "2":
            print("WARNING: h_gas_model = '2' seems to provide questionable results (if it works at all) - use it with caution. ")

        '''Initialise variables and arrays'''
        #To keep track of any coolant boiling
        boil_off_position = None
        too_low_pressure = False

        #Discretisation of the nozzle
        discretised_x = np.linspace(self.geometry.x_max, self.geometry.x_min, number_of_points) #Run from the back end (the nozzle exit) to the front (chamber)
        dx = discretised_x[0] - discretised_x[1]

        #Calculation of coolant channel length per "section"
        channel_length = self.channel_geometry(number_of_sections=number_of_points)     #number_of_sections must be equal to number_of_points

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


        '''Main loop'''
        for i in range(len(discretised_x)):
            x = discretised_x[i]
            T_gas[i] = self.T(x)

            #Get exhaust gas transport properties
            mu_gas[i] = self.exhaust_transport.mu(T = T_gas[i], p = self.p(x))
            k_gas[i] = self.exhaust_transport.k(T = T_gas[i], p = self.p(x))
            Pr_gas[i] = self.exhaust_transport.Pr(T = T_gas[i], p = self.p(x))

            if self.has_cooling_jacket and self.cooling_jacket.xs[0] <= x <= self.cooling_jacket.xs[1]:
                #Gas side heat transfer coefficient
                if h_gas_model == "1":
                    h_gas[i] = cool.h_gas_1(2*self.y(x),
                                            self.M(x),
                                            T_gas[i],
                                            self.rho(x),
                                            self.perfect_gas.gamma,
                                            self.perfect_gas.R,
                                            mu_gas[i],
                                            k_gas[i],
                                            Pr_gas[i])

                elif h_gas_model == "2":
                    #We need the previous wall temperature to use h_gas_3. If we're on the first step, then just use h_gas_1()
                    if i == 0:
                        h_gas[i] = cool.h_gas_1(2*self.y(x),
                                                self.M(x),
                                                T_gas[i],
                                                self.rho(x),
                                                self.perfect_gas.gamma,
                                                self.perfect_gas.R,
                                                mu_gas[i],
                                                k_gas[i],
                                                Pr_gas[i])
                    #Use h_gas_2() for all subsequent steps                            
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
                        mu_am = self.exhaust_transport.mu(T = T_am, p = p_inf)
                        rho_am = p_inf/(R*T_am)                                 #p = rho R T - pressure is roughly uniform across the boundary layer so p_inf ~= p_wall

                        #Stagnation properties
                        p0 = self.chamber_conditions.p0
                        T0 = self.chamber_conditions.T0
                        mu0 = self.exhaust_transport.mu(T =  T0, p = p0)

                        h_gas[i] = cool.h_gas_2(D, cp_inf, mu_inf, Pr_inf, rho_inf, v_inf, rho_am, mu_am, mu0)

                elif h_gas_model == "3":
                    #We need the previous wall temperature to use h_gas_3. If we're on the first step, then just use h_gas_1()
                    if i == 0:
                        h_gas[i] = cool.h_gas_1(2*self.y(x),
                                                self.M(x),
                                                T_gas[i],
                                                self.rho(x),
                                                self.perfect_gas.gamma,
                                                self.perfect_gas.R,
                                                mu_gas[i],
                                                k_gas[i],
                                                Pr_gas[i])

                    #Use h_gas_3() for all subsequent steps
                    else:
                        h_gas[i] = cool.h_gas_3(self.c_star,
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
                    raise AttributeError(f"Could not find the h_gas_model '{h_gas_model}'. Try '1', '2' or '3'.")

                #Calculate the current coolant temperature
                if i == 0:
                    T_coolant[i] = self.cooling_jacket.inlet_T
                    p0_coolant[i] = self.cooling_jacket.inlet_p0
                    p_coolant[i] = p0_coolant[i] - self.Q_coolant(T=T_coolant[i], p=p0_coolant[i], x=x, y=self.y(x))

                else:
                    #Increase in coolant temperature, q*dx = mdot*Cp*dT
                    T_coolant[i] = T_coolant[i-1] + (q_dot[i-1]*dx)/(self.cooling_jacket.mdot_coolant*cp_coolant[i-1]) 

                    #Pressure drop in coolant channel
                    friction_factor = self.coolant_friction_factor(T=T_coolant[i], p=p_coolant[i-1], x=x, y=self.y(x))
                    p0_coolant[i] = p0_coolant[i-1] - self.coolant_p0_drop(friction_factor, dl=channel_length[i-1], T=T_coolant[i], p=p_coolant[i-1], x=x, y=self.y(x))
                    p_coolant[i] = p0_coolant[i] - self.Q_coolant(T=T_coolant[i], p=p_coolant[i-1], x=x, y=self.y(x)) # Update static pressure of coolant

                    if too_low_pressure == False and p0_coolant[i] < self.chamber_conditions.p0:
                        too_low_pressure = True
                        print(f"Coolant stagnation pressure dropped below chamber pressure ({self.chamber_conditions.p0/1e5} bar) at x = {x}, the coolant would not flow in real life.")
                    
                    if p0_coolant[i] < 0:
                        raise ValueError("Coolant stagnation pressure dropped below 0 bar - your coolant velocities may be too high.")

                #Update coolant heat capacity, transport properties and velocity
                Pr_coolant[i] = self.cooling_jacket.coolant_transport.Pr(T = T_coolant[i], p = p_coolant[i])
                cp_coolant[i] = self.cooling_jacket.coolant_transport.cp(T = T_coolant[i], p = p_coolant[i])
                mu_coolant[i] = self.cooling_jacket.coolant_transport.mu(T = T_coolant[i], p = p_coolant[i])
                k_coolant[i] = self.cooling_jacket.coolant_transport.k(T = T_coolant[i], p = p_coolant[i])
                rho_coolant[i] = self.cooling_jacket.coolant_transport.rho(T = T_coolant[i], p = p_coolant[i])
                v_coolant[i] = self.cooling_jacket.coolant_velocity(rho_coolant[i], x=x, y = self.y(x, up_to = "wall in"))

                #Coolant side heat transfer coefficient
                if h_coolant_model == "1":
                    h_coolant[i] = cool.h_coolant_1(self.cooling_jacket.A(x=x, y=self.y(x=x, up_to = "wall in")), 
                                                    self.cooling_jacket.D(x=x, y=self.y(x=x, up_to = "wall in")), 
                                                    self.cooling_jacket.mdot_coolant, 
                                                    mu_coolant[i], 
                                                    k_coolant[i], 
                                                    cp_coolant[i], 
                                                    rho_coolant[i])

                elif h_coolant_model == "2":
                    #Use '3' for the first step (model '2' relies on the wall temperature, which hasn't yet been calculated for the first step).
                    if i == 0:
                        h_coolant[i] = cool.h_coolant_3(rho_coolant[i], 
                                                        v_coolant[i], 
                                                        self.cooling_jacket.D(x=x, y=self.y(x=x, up_to = "wall in")), 
                                                        mu_coolant[i], 
                                                        Pr_coolant[i], 
                                                        k_coolant[i])

                    #Use model '2' for every other step.
                    else:
                        h_coolant[i] = cool.h_coolant_2(rho_coolant[i], 
                                                        v_coolant[i], 
                                                        self.cooling_jacket.D(x=x, y=self.y(x=x, up_to = "wall in")), 
                                                        mu_coolant[i], 
                                                        self.cooling_jacket.coolant_transport.mu(T = T_wall_outer[i-1], p = p_coolant[i]), 
                                                        Pr_coolant[i], 
                                                        k_coolant[i])

                elif h_coolant_model == "3":
                    h_coolant[i] = cool.h_coolant_3(rho_coolant[i], 
                                                    v_coolant[i], 
                                                    self.cooling_jacket.D(x=x, y=self.y(x=x, up_to = "wall in")), 
                                                    mu_coolant[i], 
                                                    Pr_coolant[i], 
                                                    k_coolant[i])

                else:
                    raise AttributeError(f"Could not find the h_coolant_model '{h_coolant_model}'")
                
                #Check for coolant boil off 
                if boil_off_position == None and self.cooling_jacket.coolant_transport.check_liquid(T = T_coolant[i], p = p_coolant[i]) == False:
                    print(f"WARNING: Coolant boiled off at x = {x} m")
                    boil_off_position = x

                #Get thermal circuit properties
                if self.has_ablative and self.ablative.xs[0] <= x <= self.ablative.xs[1]:
                    #Thermal circuit
                    q_dot[i], R_gas[i], R_ablative[i], R_wall[i], R_coolant[i] = self.regen_ablative_thermal_circuit(self.y(x), 
                                                                                                            h_gas[i], 
                                                                                                            h_coolant[i], 
                                                                                                            self.ablative.wall_material, 
                                                                                                            self.thickness(x, layer = 'wall'), 
                                                                                                            T_gas[i], 
                                                                                                            T_coolant[i], 
                                                                                                            self.ablative.ablative_material, 
                                                                                                            self.thickness(x, layer = 'ablative'))
                    
                    #Calculate wall temperatures using the thermal circuit idea
                    T_ablative_inner[i] = T_gas[i] - q_dot[i]*R_gas[i]
                    T_wall_inner[i] = T_ablative_inner[i] - q_dot[i]*R_ablative[i]
                    T_wall_outer[i] = T_wall_inner[i] - q_dot[i]*R_wall[i]
                
                else:
                    q_dot[i], R_gas[i], R_wall[i], R_coolant[i] = self.regen_thermal_circuit(self.y(x), 
                                                                                    h_gas[i], 
                                                                                    h_coolant[i], 
                                                                                    self.cooling_jacket.inner_wall, 
                                                                                    self.thickness(x, layer = 'wall'), 
                                                                                    T_gas[i], 
                                                                                    T_coolant[i])

                    #Calculate temperatures
                    T_wall_inner[i] = T_gas[i] - q_dot[i]*R_gas[i]
                    T_wall_outer[i] = T_wall_inner[i] - q_dot[i]*R_wall[i]
            else:
                T_wall_inner[i] = T_gas[i]
                T_wall_outer[i] = T_gas[i]

        #Not sure if the Sieder-Tate equation is valid with your coolant above the boiling temperature at the wall.
        if h_coolant_model == '2' and self.cooling_jacket.coolant_transport.check_liquid(T = np.amax(T_wall_outer[i]), p = p_coolant[-1]) == False:
            print("Coolant temperature at the wall was above its boiling point when using the Sieder-Tate equation (h_coolant_model = '2') - results should be used with caution.")

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
                "boil_off_position" : boil_off_position}

        #Export a .JSON file if required
        if to_json != False:
            with open(to_json, "w+") as write_file:
                json.dump(output_dict, write_file)
                print("Exported JSON data to '{}'".format(to_json))

        return output_dict

    def transient_heating_analysis(self, number_of_points=1000, dt = 0.1, t_max = 100, wall_starting_T = 298.15, h_gas_model = "1", to_json = "heating_output.json"):
        """This is used exclusive for pure ablative cooling, without any regenerative cooling jacket.

        Note:
            This function is outdated and does not incorporate many new features that have been added to Bamboo.

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
            self.exhaust_transport
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
                                            self.exhaust_transport.mu(T = T_gas[i, j], p = self.p(x)),
                                            self.exhaust_transport.k(T = T_gas[i, j], p = self.p(x)),
                                            self.exhaust_transport.Pr(T = T_gas[i, j], p = self.p(x)))

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
                    mu_inf = self.exhaust_transport.mu(T = T_gas[i, j], p = p_inf)
                    Pr_inf = self.exhaust_transport.Pr(T = T_gas[i, j], p = p_inf)
                    cp_inf = self.perfect_gas.cp

                    #Properties at arithmetic mean of T_wall and T_inf
                    T_am = (T_inf + T_ablative_inner[i, j-1]) / 2
                    mu_am = self.exhaust_transport.mu(T = T_am, p = p_inf)
                    rho_am = p_inf/(R*T_am)                                 #p = rho R T - pressure is roughly uniform across the boundary layer so p_inf ~= p_wall

                    #Stagnation properties
                    p0 = self.chamber_conditions.p0
                    T0 = self.chamber_conditions.T0
                    mu0 = self.exhaust_transport.mu(T =  T0, p = p0)

                    h_gas[i, j] = cool.h_gas_2(D, cp_inf, mu_inf, Pr_inf, rho_inf, v_inf, rho_am, mu_am, mu0)

                elif h_gas_model == "3":
                    h_gas[i, j] = cool.h_gas_3(self.c_star,
                                            self.nozzle.At, 
                                            self.A(x), 
                                            self.chamber_conditions.p0, 
                                            self.chamber_conditions.T0, 
                                            self.M(x), 
                                            T_ablative_inner[i, j-1], 
                                            self.exhaust_transport.mu(T = T_gas[i, j], p = self.p(x)), 
                                            self.perfect_gas.cp, 
                                            self.perfect_gas.gamma, 
                                            self.exhaust_transport.Pr(T = T_gas[i, j], p = self.p(x)))

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


    def run_stress_analysis(self, heating_result, condition="steady", **kwargs):
        """Perform stress analysis on the liner, using a cooling result.
           Results should be taken only as a first approximation of some key stresses.

        Args:
            heating_result (dict): Requires a heating analysis result to compute stress.
            condition (str, optional): Engine state for analysis. Options are "steady", or "transient". Defaults to "steady".

        Keyword Args:
            T_amb (float, optional): For transient analysis, the ambient temperature can be overriden from the default, 283 K.
                                     This is used for calculating thermal expansions, so it's really the zero thermal stress
                                     temperature, presumably the temperature at which the engine was assembled.

        Returns:
            dict: Results of the stress simulation. Contains the following dictionary keys (all are arrays):
                For a steady state analysis: 
                - "thermal_stress : Stress induced in the inner liner due to temperature difference, chamber to coolant side, for each x value (Pa).
                - "tadjusted_yield : Yield stress of the inner wall material, corrected for the chamber side temperature (worst case) (Pa).
                - "deltaT_wall" : Inner liner temperature difference, chamber side - coolant side (K).
                - "stress_inner_hoop_steady" : Hoop stress of inner liner due to coolant static pressure in jacket after ignition (Pa).
                - "stress_outer_hoop" : Hoop stress of outer liner due to coolant static pressure (same before and after ignition) (Pa).
                For a transient analysis:
                - "stress_inner_hoop_transient" : Hoop stress of inner liner due to coolant static pressure in jacket prior to ignition (0 chamber pressure) (Pa).
                - "stress_inner_IE" : Stress induced in inner liner as it is heated but constrained by cold outer liner (Pa).
                - "stress_outer_IE : Stress induced in outer liner by expanding inner liner (Pa).
        """
        length = len(heating_result["x"])
        wall_stress = np.zeros(length)
        wall_deltaT = np.zeros(length)
        tadjusted_yield = np.zeros(length)
        mat_in = self.cooling_jacket.inner_wall
        mat_out = self.cooling_jacket.outer_wall

        E1 = self.cooling_jacket.inner_wall.E
        E2 = self.cooling_jacket.outer_wall.E
        t1_hoop = self.map_thickness_profile(self.geometry.inner_wall_thickness, length)
        t1 = t1_hoop + self.cooling_jacket.channel_height*self.cooling_jacket.blockage_ratio
        # The blockage ratio is used to scale the contribution of the ribs to the
        # effective total thickness of the inner liner (wider ribs = greater blockage ratio)
        t2 = self.map_thickness_profile(self.geometry.outer_wall_thickness, length)

        # Geometry calculations used for non-thermal stresses;
        # if there is an ablative, the inner liner radius will be
        # constant as the nozzle geometry is created with this instead of
        # the cooling jacket.
        if self.has_ablative is True or self.cooling_jacket.has_ablative is True:
            if self.ablative.wall_material != self.cooling_jacket.inner_wall:
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
