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
    def __init__(self, combustion_chamber, nozzle, chamber_length, wall_thickness, geometry="auto"):
        self.combustion_chamber = combustion_chamber
        self.nozzle = nozzle
        self.chamber_length = chamber_length
        self.wall_thickness = wall_thickness
        self.geometry = geometry


        if self.geometry == "auto":
            #Use the system defined in Reference [1]
            self.theta_curved_converging_start = -3*np.pi/4
            self.x_curved_converging_start = 1.5*self.nozzle.Rt*np.cos(self.theta_curved_converging_start)
            self.y_curved_converging_start = 1.5*self.nozzle.Rt*np.sin(self.theta_curved_converging_start) + 1.5*self.nozzle.Rt + self.nozzle.Rt

            #Find the gradient where the curved converging bit starts
            dxdtheta_curved_converging_start = -1.5*self.nozzle.Rt*np.sin(self.theta_curved_converging_start)
            self.dydx_curved_converging_start = -1.5*self.nozzle.Rt*np.cos(self.theta_curved_converging_start)/dxdtheta_curved_converging_start

            #Find the x-position where we reach the combustion chamber radius
            self.x_chamber_end = self.x_curved_converging_start - (self.combustion_chamber.R - self.y_curved_converging_start)/self.dydx_curved_converging_start

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
                    return self.combustion_chamber.R

                #Inside the converging section
                elif x >= self.x_chamber_end:
                    return np.interp(x, [self.x_chamber_end, self.x_curved_converging_start], [self.combustion_chamber.R, self.y_curved_converging_start])

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
    def __init__(self, k_wall, channel_width, channel_height, inlet_T, inlet_p0, coolant, mdot_coolant, channel_shape = "rectangle", configuration = "spiral"):
        self.k_wall = k_wall
        self.channel_width = channel_width
        self.channel_height = channel_height
        self.coolant = coolant          #thermo library Chemical
        self.mdot_coolant = mdot_coolant
        self.inlet_T = inlet_T
        self.inlet_p0 = inlet_p0
        self.channel_shape = channel_shape
        self.configuration = configuration
        
        if self.channel_shape == "rectangle":
            #Page 317 of RPE 7th Edition
            self.perimeter = 2*channel_width + 2*channel_height
            self.flow_area = channel_width*channel_height
            self.hydraulic_radius = self.flow_area/self.perimeter
            self.equivelant_diameter = 4*self.hydraulic_radius

    def A(self, x=None):
        return self.flow_area
    
    def D(self, x=None):
        return self.equivelant_diameter

class EngineWithCooling:
    def __init__(self, engine_geometry, cooling_jacket, gas, thermo_gas):
        self.geometry = engine_geometry
        self.cooling_jacket = cooling_jacket
        self.gas = gas
        self.thermo_gas = thermo_gas

    def M(self, x):
        #Exhaust gas Mach number
        #If we're at the throat M=1 by default:
        if x==0:
            return 1.00

        #If we're not at the throat:
        else:
            def func_to_solve(Mach):
                return self.geometry.combustion_chamber.mdot*(self.gas.cp*self.geometry.combustion_chamber.T0)**0.5 / (self.geometry.A(x)*self.geometry.combustion_chamber.p0) - bam.m_bar(Mach, self.gas.gamma)
            
            if x > 0:
                Mach = scipy.optimize.root_scalar(func_to_solve, bracket = [1,300], x0 = 1).root
            else:
                Mach = scipy.optimize.root_scalar(func_to_solve, bracket = [0,1], x0 = 0.5).root

            return Mach

    def T(self, x):
        #Exhaust gas temperature
        return bam.T(self.geometry.combustion_chamber.T0, self.M(x), self.gas.gamma)

    def p(self, x):
        return bam.p(self.geometry.combustion_chamber.p0, self.M(x), self.gas.gamma)

    def rho(self, x):
        #Exhaust gas density
        #p = rhoRT for an ideal gas, so rho = p/RT
        return self.p(x)/(self.T(x)*self.gas.R)

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

    def coolant_velocity(self, x, rho):
        #rho is the coolant density here!
        return self.cooling_jacket.mdot_coolant/(rho * self.cooling_jacket.A(x))

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
        return 0.026*(self.rho(x)*self.M(x)*(self.gas.gamma*self.gas.R*self.T(x))**0.5)**0.8 / (self.cooling_jacket.D(x))**0.2 * Pr**0.4 * k/(mu**0.8)

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
        return 0.023*c_bar*(self.geometry.combustion_chamber.mdot/self.cooling_jacket.A(x))*(self.cooling_jacket.D(x)*self.coolant_velocity(x, rho)*rho/mu)**(-0.2)*(mu*c_bar/k)**(-2/3) 

    def thermal_circuit(self, x, h_gas, h_coolant, k_wall, T_gas, T_coolant):
        """
        q is per unit length along the nozzle wall (axially) - positive when heat is flowing to the coolant.
        Uses the idea of thermal circuits and resistances - we have three resistors in series.

        Args:
            x (float): x position (m)
            h_gas (float): Gas side convective heat transfer coefficient
            h_coolant (float): Coolant side convective heat transfer coefficient
            k_wall (float): Wall thermal conductivity
            T_gas (float): Free stream gas temperature (K)
            T_coolant (float): Coolant temperature (K)

        Returns:
            float, float, float, float: q_dot, R_gas, R_wall, R_coolant
        """

        r = self.geometry.y(x)
        
        r_out = r + self.geometry.wall_thickness
        r_in = r - self.geometry.wall_thickness

        A_in = 2*np.pi*r_out    #Inner area per unit length (i.e. just the inner circumference)
        A_out = 2*np.pi*r_in    #Outer area per unit length (i.e. just the outer circumference)

        R_gas = 1/(h_gas*A_in)
        R_wall = np.log(r_out/r_in)/(2*np.pi*k_wall)
        R_coolant = 1/(h_coolant*A_out)

        q_dot = (T_gas - T_coolant)/(R_gas + R_wall + R_coolant)

        return q_dot, R_gas, R_wall, R_coolant

    def run_heating_analysis(self, number_of_points=1000):
        discretised_x = np.linspace(self.geometry.x_min, self.geometry.x_max, number_of_points)
        dx = discretised_x[1] - discretised_x[0]

        T_wall_inner = np.zeros(len(discretised_x))
        T_wall_outer = np.zeros(len(discretised_x))
        T_coolant = np.zeros(len(discretised_x))
        T_gas = np.zeros(len(discretised_x))
        q_dot = np.zeros(len(discretised_x))

        coolant = self.cooling_jacket.coolant
        exhaust_gas = self.thermo_gas

        for i in range(len(discretised_x)):
            x = discretised_x[i]
            
            if i == 0:
                T_gas[i] = self.T(x)
                p_gas = self.p(x)
                T_coolant[i] = self.cooling_jacket.inlet_T

                coolant.calculate(T = T_coolant[i], P = self.cooling_jacket.inlet_p0)
                exhaust_gas.calculate(T = T_gas[i], P = p_gas)

                h_gas = self.h_gas(x, exhaust_gas.mu, exhaust_gas.k, exhaust_gas.Pr)
                h_coolant = self.h_coolant(x, coolant.mu, coolant.k, coolant.Cp, coolant.rho)

                q_dot[i], R_gas, R_wall, R_coolant = self.thermal_circuit(x, h_gas, h_coolant, self.cooling_jacket.k_wall, T_gas[i], T_coolant[i])

                T_wall_inner[i] = T_gas[i] - q_dot[i]*R_gas
                T_wall_outer[i] = T_wall_inner[i] - q_dot[i]*R_wall

            else:    
                T_gas[i] = self.T(x)
                p_gas = self.p(x)
                T_coolant[i] = T_coolant[i-1] + (q_dot[i-1]*dx)/(self.cooling_jacket.mdot_coolant*self.cooling_jacket.coolant.Cp)    #Increase in coolant temperature, q*dx = mdot*Cp*dT

                coolant.calculate(T = T_coolant[i], P = self.cooling_jacket.inlet_p0)
                exhaust_gas.calculate(T = T_gas[i], P = p_gas)

                h_gas = self.h_gas(x, exhaust_gas.mu, exhaust_gas.k, exhaust_gas.Pr)
                h_coolant = self.h_coolant(x, coolant.mu, coolant.k, coolant.Cp, coolant.rho)

                q_dot[i], R_gas, R_wall, R_coolant = self.thermal_circuit(x, h_gas, h_coolant, self.cooling_jacket.k_wall, T_gas[i], T_coolant[i])

                T_wall_inner[i] = T_gas[i] - q_dot[i]*R_gas
                T_wall_outer[i] = T_wall_inner[i] - q_dot[i]*R_wall
            
        return {"x" : discretised_x,
                "T_wall_inner" : T_wall_inner,
                "T_wall_outer" : T_wall_outer,
                "T_coolant" : T_coolant,
                "T_gas" : T_gas,
                "q_dot" : q_dot}