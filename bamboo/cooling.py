'''
Extra tools for modelling the cooling system of a liquid rocket engine.

Room for improvement:
    - My equation for h_gas is the less accurate version, with the Bartz correction factors (this was just to avoid needing the extra parameters for the Bartz equation)
    - The EngineWithCooling.rho() function calculates rho by doing p/RT, but it would probably be faster to just use isentropic compressible flow relations.

References:
    [1] - The Thrust Optimised Parabolic nozzle, AspireSpace, http://www.aspirespace.org.uk/downloads/Thrust%20optimised%20parabolic%20nozzle.pdf  
    [2] - Rocket Propulsion Elements, 7th Edition
'''

import bamboo as bam
import numpy as np
import matplotlib.pyplot as plt
import scipy

'''Constants'''
SIGMA = 5.670374419e-8      #Stefan-Boltzmann constant (W/m^2/K^4)

'''Functions'''
def black_body(T):
    """Get the black body radiation emitted over a hemisphere, at a given temperature.

    Args:
        T (float): Temperature of the body (K)

    Returns:
        float: Radiative heat transfer rate, per unit emitting area on the body (W/m2)
    """
    return SIGMA*T**4

'''Classes'''
class EngineGeometry:
    def __init__(self, nozzle, chamber_length, chamber_area, wall_thickness, geometry="auto"):
        self.nozzle = nozzle
        self.chamber_length = chamber_length
        self.chamber_area = chamber_area
        self.chamber_radius = (chamber_area/np.pi)**0.5 
        self.wall_thickness = wall_thickness
        self.geometry = geometry

        if self.nozzle.At > self.chamber_area:
            raise ValueError(f"The combustion chamber area {self.chamber_area} m^2 is smaller than the throat area {self.nozzle.At} m^2.")

        if self.geometry == "auto":
            #Use the system defined in Reference [1] - mostly using Eqns (4)
            #Make sure we cap the size of the converging section to the radius of the combustion chamber.
            chamber_radius = (self.chamber_area/np.pi)**0.5
            theta_min = -np.pi - np.arcsin((chamber_radius - self.nozzle.Rt - 1.5*self.nozzle.Rt) / (1.5*self.nozzle.Rt)) 
            if theta_min > -3*np.pi/4:
                self.theta_curved_converging_start = theta_min
            else:
                self.theta_curved_converging_start = -3*np.pi/4

            #Find key properties for the converging section
            self.x_curved_converging_start = 1.5*self.nozzle.Rt*np.cos(self.theta_curved_converging_start)
            self.y_curved_converging_start = 1.5*self.nozzle.Rt*np.sin(self.theta_curved_converging_start) + 1.5*self.nozzle.Rt + self.nozzle.Rt

            #Find the gradient where the curved converging bit starts
            dxdtheta_curved_converging_start = -1.5*self.nozzle.Rt*np.sin(self.theta_curved_converging_start)
            self.dydx_curved_converging_start = -1.5*self.nozzle.Rt*np.cos(self.theta_curved_converging_start)/dxdtheta_curved_converging_start

            #Find the x-position where we reach the combustion chamber radius
            self.x_chamber_end = self.x_curved_converging_start - (self.chamber_radius - self.y_curved_converging_start)/self.dydx_curved_converging_start

            #Start and end points of the engine
            self.x_min = self.x_chamber_end - self.chamber_length
            self.x_max = self.nozzle.length

    def y(self, x):
        if self.geometry == "auto":
            #Curved converging section
            if x < 0 and x > self.x_curved_converging_start:
                theta = -np.arccos(x/(1.5*self.nozzle.Rt))
                return 1.5*self.nozzle.Rt*np.sin(theta) + 1.5*self.nozzle.Rt + self.nozzle.Rt

            #Before the curved part of the converging section
            elif x <= self.x_curved_converging_start:
                #Inside the chamber
                if x < self.x_chamber_end and x >= self.x_min:
                    return self.chamber_radius

                #Inside the converging section
                elif x >= self.x_chamber_end:
                    return np.interp(x, [self.x_chamber_end, self.x_curved_converging_start], [self.chamber_radius, self.y_curved_converging_start])

                #Outside of the engine
                else:
                    return ValueError(f"x is beyond the front of the engine. You tried to input {x} but the minimum value you're allowed is {self.x_chamber_end - self.chamber_length}")
            
            #In the diverging section of the nozzle
            elif x >= 0:
                return self.nozzle.y(x)

    def A(self, x):
        return np.pi*self.y(x)**2

    def plot_geometry(self, number_of_points = 1000):
            x = np.linspace(self.x_min, self.x_max, number_of_points)
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

class CoolingJacket:
    """Cooling jacket parameters.

    Args:
        inner_wall (material): Inner wall material
        inlet_T (float): Inlet coolant temperature (K)
        inlet_p0 (float): Inlet coolant stagnation pressure (Pa)
        thermo_coolant (thermo.chemical.Chemical or thermo.mixture.Mixture): Used to get physical properties of the coolant.
        mdot_coolant (float): Coolant mass flow rate (kg/s)
        channel_shape (str, optional): Options include 'rectangle', 'semi-circle', and 'custom'. Defaults to "rectangle".
        configuration (str, optional): Options include 'spiral'. Defaults to "spiral".
        rectangle_width (float, optional): If using channel_shape = 'rectangle', this is the height of the rectangles (in the radial direction). Defaults to None.
        rectangle_height (float, optional): If using channel_shape = 'rectangle, this is the width of the rectangles (in the hoopwise direction). Defaults to None.
        circle_diameter (float, optional): If using channel_shape = 'semi-circle', this is the diameter of the semi circle. Defaults to None.
        custom_hydraulic_diameter (float, optional) : If using channel_shape = 'custom', this is the hydraulic diameter you want to use. Defautls to None.
        custom_flow_area (float, optional) : If using channel_shape = 'custom', this is the flow you want to use. Defaults to None.
    """
    def __init__(self, inner_wall, inlet_T, inlet_p0, thermo_coolant, mdot_coolant, channel_shape = "rectangle", configuration = "spiral", 
                 rectangle_width = None, rectangle_height = None,
                 circle_diameter = None,
                 custom_hydraulic_diameter = None, custom_flow_area = None):
        self.inner_wall = inner_wall
        self.thermo_coolant = thermo_coolant          #thermo library Chemical
        self.mdot_coolant = mdot_coolant
        self.inlet_T = inlet_T
        self.inlet_p0 = inlet_p0
        self.channel_shape = channel_shape
        self.configuration = configuration
        
        if self.channel_shape == "rectangle":
            #Page 317 of RPE 7th Edition
            self.rectangle_width = rectangle_width
            self.rectangle_height = rectangle_height
            self.perimeter = 2*rectangle_width + 2*rectangle_height
            self.flow_area = rectangle_width*rectangle_height
            self.hydraulic_radius = self.flow_area/self.perimeter
            self.hydraulic_diameter = 4*self.hydraulic_radius

        if self.channel_shape == "semi-circle":
            self.circle_diameter = circle_diameter
            self.perimeter = circle_diameter + np.pi*circle_diameter/2
            self.flow_area = np.pi*circle_diameter**2/8
            self.hydraulic_radius = self.flow_area/self.perimeter
            self.hydraulic_diameter = 4*self.hydraulic_radius

        if self.channel_shape == "custom":
            self.flow_area = custom_flow_area
            self.hydraulic_diameter = custom_hydraulic_diameter


    def A(self, x=None):
        return self.flow_area
    
    def D(self, x=None):
        return self.hydraulic_diameter

class Material:
    def __init__(self, E, sigma_y, poisson, alpha, k):
        self.E = E # Young's modulus
        self.sigma_y = sigma_y # 0.2% yield stress
        self.poisson = poisson
        self.alpha = alpha # Thermal expansion coeff
        self.k = k # Thermal conductivity

    def performance_thermal(self, poisson, alpha, k):
        # Performance coefficient for thermal stress, higher is better
        return (1 - poisson)*k/alpha

class EngineWithCooling:
    def __init__(self, chamber_conditions, geometry, cooling_jacket, perfect_gas, thermo_gas):
        self.chamber_conditions = chamber_conditions
        self.geometry = geometry
        self.cooling_jacket = cooling_jacket
        self.perfect_gas = perfect_gas
        self.thermo_gas = thermo_gas
        #self.c_star = self.geometry.chamber_conditions.p0 * self.geometry.nozzle.At / self.geometry.chamber_conditions.mdot

    def M(self, x):
        #Exhaust gas Mach number
        #If we're at the throat M=1 by default:
        if x==0:
            return 1.00

        #If we're not at the throat:
        else:
            def func_to_solve(Mach):
                return self.chamber_conditions.mdot*(self.perfect_gas.cp*self.chamber_conditions.T0)**0.5 / (self.geometry.A(x)*self.chamber_conditions.p0) - bam.m_bar(Mach, self.perfect_gas.gamma)
            
            if x > 0:
                Mach = scipy.optimize.root_scalar(func_to_solve, bracket = [1,300], x0 = 1).root
            else:
                Mach = scipy.optimize.root_scalar(func_to_solve, bracket = [0,1], x0 = 0.5).root

            return Mach

    def T(self, x):
        #Exhaust gas temperature
        return bam.T(self.chamber_conditions.T0, self.M(x), self.perfect_gas.gamma)

    def p(self, x):
        return bam.p(self.chamber_conditions.p0, self.M(x), self.perfect_gas.gamma)

    def rho(self, x):
        #Exhaust gas density
        #p = rhoRT for an ideal gas, so rho = p/RT
        return self.p(x)/(self.T(x)*self.perfect_gas.R)

    def show_gas_temperature(self, number_of_points=1000):
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

            plt.show()

    def show_gas_mach(self, number_of_points=1000):
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

            plt.show()

    def coolant_velocity(self, x, rho_coolant):
        return self.cooling_jacket.mdot_coolant/(rho_coolant * self.cooling_jacket.A(x))

    def h_gas(self, x, mu, k, Pr):
        """Get the convective heat transfer coefficient on the gas side.
        Uses Eqn (8-22) on page 312 or RPE 7th edition - would be better to use the Bartz version if possible (Eqn (8-23))

        Args:
            x (float): x-position (m)
            mu (float): Absolute viscosity of the exhaust gas
            k (float): Thermal conductivity of the exhaust gas
            Pr (float): Prandtl number of the exhaust gas


        Returns:
            float: Gas side convective heat transfer coefficient
        """
        M = self.M(x)
        T = self.T(x)
        rho = self.rho(x)
        gamma = self.perfect_gas.gamma
        R = self.perfect_gas.R

        v = M * (gamma*R*T)**0.5    #Gas velocity
        D = 2*self.geometry.y(x)    #Flow diameter

        return 0.026 * (rho*v)**0.8 / (D**0.2) * (Pr**0.4) * k/(mu**0.8)

    def h_gas_bartz_1(self, D, cp_inf, mu_inf, Pr_inf, rho_inf, v_inf, rho_am, mu_am, mu0):
        """Equation (8-23) from page 312 of RPE 7th edition. 'am' refers to the gas being at the 'arithmetic mean' of the wall and freestream temperatures.

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
        """

        return (0.026/D**0.2) * (cp_inf*mu_inf**0.2)/(Pr_inf**0.6) * (rho_inf * v_inf)**0.8 * (rho_am/rho_inf) * (mu_am/mu0)**0.2

    def h_gas_bartz_2(self, mu, cp, Pr, M, A, Tw):
        """Altnerative equation for Bartz.

        Args:
            mu (float): Absolute viscosity of the gas freestream.
            cp (float): Specific heat at constant pressure for the gas freestream.
            Pr (float): Prandtl number for the gas freestream.
            M (float): Mach number in the gas freestream.
            A (float): Flow area of the gas.
            Tw (float): Gas temperature at the wall.
        """
        c_star = self.chamber_conditions.p0 * self.geometry.nozzle.At / self.chamber_conditions.mdot
        Dt = 2*self.geometry.nozzle.Rt
        At = self.geometry.nozzle.At
        pc = self.chamber_conditions.p0
        Tc = self.chamber_conditions.T0

        gamma = self.perfect_gas.gamma

        sigma = (0.5 * (Tw/Tc) * (1 + (gamma-1)/2 * M**2) + 0.5)**0.68 * (1 + (gamma-1)/2 * M**2)**(-0.12)
        
        return (0.026)/(Dt**0.2) * (mu**0.2*cp/Pr**0.6) * (pc/c_star)**0.8 * (At/A)**0.9 * sigma

    def h_coolant(self, x, mu, k, c_bar, rho):
        """Get the convective heat transfer coefficient for the coolant side.
        Uses the equation from page 317 of RPE 7th edition.

        Args:
            x (float): x-position (m)
            mu (float): Absolute viscosity of coolant 
            k (float): Thermal conductivity of coolant
            c_bar(float): Average specific heat capacity of coolant
            rho (float): Density of coolant (kg/m3)

        Returns:
            float: Coolant side convective heat transfer coefficient
        """
        mdot = self.cooling_jacket.mdot_coolant
        A = self.cooling_jacket.A(x)
        D = self.cooling_jacket.D(x)
        v = self.coolant_velocity(x, rho)
        
        return 0.023*c_bar * (mdot/A) * (D*v*rho/mu)**(-0.2) * (mu*c_bar/k)**(-2/3)

    def thermal_circuit(self, x, h_gas, h_coolant, inner_wall, T_gas, T_coolant):
        """
        q is per unit length along the nozzle wall (axially) - positive when heat is flowing to the coolant.
        Uses the idea of thermal circuits and resistances - we have three resistors in series.

        Args:
            x (float): x position (m)
            h_gas (float): Gas side convective heat transfer coefficient
            h_coolant (float): Coolant side convective heat transfer coefficient
            inner_wall (material): Inner wall material, needed for thermal conductivity
            T_gas (float): Free stream gas temperature (K)
            T_coolant (float): Coolant temperature (K)

        Returns:
            float, float, float, float: q_dot, R_gas, R_wall, R_coolant
        """

        r = self.geometry.y(x)
        
        r_out = r + self.geometry.wall_thickness
        r_in = r 

        A_in = 2*np.pi*r_out    #Inner area per unit length (i.e. just the inner circumference)
        A_out = 2*np.pi*r_in    #Outer area per unit length (i.e. just the outer circumference)

        R_gas = 1/(h_gas*A_in)
        R_wall = np.log(r_out/r_in)/(2*np.pi*inner_wall.k)
        R_coolant = 1/(h_coolant*A_out)

        q_dot = (T_gas - T_coolant)/(R_gas + R_wall + R_coolant)

        return q_dot, R_gas, R_wall, R_coolant

    def run_heating_analysis(self, number_of_points=1000, h_gas_model = "standard"):
        """Run a simulation of the engine cooling system to get wall temperatures, coolant temperatures, etc.

        Args:
            number_of_points (int, optional): Number of discrete points to divide the engine into. Defaults to 1000.
            h_gas_model (str, optional): Equation to use for the gas side convective heat transfer coefficients. Options are 'standard' and 'bartz 1', 'bartz 2'. Defaults to "standard".

        Returns:
            dict: Results of the simulation.
        """

        '''Initialise variables and arrays'''
        #To keep track of any coolant boiling
        boil_off_position = None
        
        #Discretisation of the nozzle
        discretised_x = np.linspace(self.geometry.x_max, self.geometry.x_min, number_of_points) #Run from the back end (the nozzle exit) to the front (chamber)
        dx = discretised_x[0] - discretised_x[1]

        #Temperatures and heat transfer rates
        T_wall_inner = np.zeros(len(discretised_x)) #Gas side wall temperature
        T_wall_outer = np.zeros(len(discretised_x)) #Coolant side wall temperature
        T_coolant = np.zeros(len(discretised_x))    #Coolant temperature
        T_gas = np.zeros(len(discretised_x))        #Freestream gas temperature
        q_dot = np.zeros(len(discretised_x))        #Heat transfer rate per unit length

        #Heat transfer rates
        h_gas = np.zeros(len(discretised_x))
        h_coolant = np.zeros(len(discretised_x))

        #Make copies of the thermo module Chemicals, so we can modify them
        coolant = self.cooling_jacket.thermo_coolant
        exhaust_gas = self.thermo_gas

        '''Main loop'''
        for i in range(len(discretised_x)):
            x = discretised_x[i]

            #Coolant side heat transfer coefficient
            coolant.calculate(T = T_coolant[i], P = self.cooling_jacket.inlet_p0)

            mu_coolant = coolant.mu
            k_coolant = coolant.k
            cp_coolant = coolant.Cp
            rho_coolant = coolant.rho

            h_coolant[i] = self.h_coolant(x, mu_coolant, k_coolant, cp_coolant, rho_coolant)

            #Check for coolant boil off
            if boil_off_position == None and coolant.phase=='g':
                print(f"WARNING: Coolant boiled off at x = {x} m")
                boil_off_position = x

            #Calculate coolant temperature
            if i == 0:
                T_coolant[i] = self.cooling_jacket.inlet_T
            else:
                T_coolant[i] = T_coolant[i-1] + (q_dot[i-1]*dx)/(self.cooling_jacket.mdot_coolant*cp_coolant)    #Increase in coolant temperature, q*dx = mdot*Cp*dT

            #Gas freestream temperatures
            T_gas[i] = self.T(x)
            p_gas = self.p(x)

            #Gas side heat transfer coefficient
            if h_gas_model == "standard":
                exhaust_gas.calculate(T = T_gas[i], P = p_gas)

                mu = exhaust_gas.mu
                k = exhaust_gas.k
                Pr = exhaust_gas.Pr

                h_gas[i] = self.h_gas(x, mu, k, Pr)

            elif h_gas_model == "bartz 1":
                gamma = self.perfect_gas.gamma
                R = self.perfect_gas.R
                D = 2*self.geometry.y(x)            #Flow diameter

                #Freestream properties
                p_inf = p_gas
                T_inf = T_gas[i]
                rho_inf = self.rho(x)
                M_inf = self.M(x)
                v_inf = M_inf * (gamma*R*T_inf)**0.5    #Gas velocity

                exhaust_gas.calculate(T = T_gas[i], P = p_gas)
                mu_inf = exhaust_gas.mu
                Pr_inf = exhaust_gas.Pr
                cp_inf = self.perfect_gas.cp

                #Properties at arithmetic mean of T_wall and T_inf
                T_am = (T_inf + T_wall_inner[i-1]) / 2

                exhaust_gas.calculate(T = T_am, P = p_gas)
                mu_am = exhaust_gas.mu
                rho_am = p_inf/(R*T_am) #p = rho R T - pressure is roughly uniform across the boundary layer so p_inf ~= p_wall

                #Stagnation properties
                p0 = self.chamber_conditions.p0
                T0 = self.chamber_conditions.T0

                exhaust_gas.calculate(T = T0, P = p0)
                mu0 = exhaust_gas.mu

                h_gas[i] = self.h_gas_bartz_1(D, cp_inf, mu_inf, Pr_inf, rho_inf, v_inf, rho_am, mu_am, mu0)

            elif h_gas_model == "bartz 2":
                M = self.M(x) 
                A = self.geometry.A(x)
                Tw = T_wall_inner[i-1]

                exhaust_gas.calculate(T = T_gas[i], P = p_gas)
                mu = exhaust_gas.mu
                cp = self.perfect_gas.cp
                Pr = exhaust_gas.Pr
                h_gas[i] = self.h_gas_bartz_2(mu, cp, Pr, M, A, Tw)

            else:
                raise AttributeError(f"Could not find the h_gas_model {h_gas_model}")
            
            #Get thermal circuit properties
            q_dot[i], R_gas, R_wall, R_coolant = self.thermal_circuit(x, h_gas[i], h_coolant[i], self.cooling_jacket.inner_wall, T_gas[i], T_coolant[i])

            #Calculate wall temperatures
            T_wall_inner[i] = T_gas[i] - q_dot[i]*R_gas
            T_wall_outer[i] = T_wall_inner[i] - q_dot[i]*R_wall

        return {"x" : discretised_x,
                "T_wall_inner" : T_wall_inner,
                "T_wall_outer" : T_wall_outer,
                "T_coolant" : T_coolant,
                "T_gas" : T_gas,
                "q_dot" : q_dot,
                "h_gas" : h_gas,
                "h_coolant" : h_coolant,
                "boil_off_position" : boil_off_position}