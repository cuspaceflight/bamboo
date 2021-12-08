"""
Main Engine class, as well as tools for specifying engine geometry and the perfect gas model used to calculate flow properties.
"""

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

import bamboo.rao
import bamboo.isen

R_BAR = 8.3144621e3         # Universal gas constant (J/K/kmol)

class PerfectGas:
    """Object to store a perfect gas model (i.e. an ideal gas with constant cp and cv). You only need to input 2 properties to fully define it.

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
            raise ValueError(f"Not enough inputs provided to fully define the PerfectGas, or you used a combination of inputs that isn't currently allowable. You must provide exactly 2 inputs, but you provided {len(kwargs)}.")

    def __repr__(self):
        return f"<nozzle.perfect_gas object> with: \ngamma = {self.gamma} \ncp = {self.cp} \nmolecular_weight = {self.molecular_weight} \nR = {self.R}"

class ChamberConditions:
    """Object for storing combustion chamber thermodynamic conditions. The mass flow rate does not have to be defined - it is fixed by the nozzle throat area.

    Args:
        p0 (float): Gas stagnation pressure (Pa).
        T0 (float): Gas stagnation temperature (K).
    """
    def __init__(self, p0, T0):
        self.p0 = p0
        self.T0 = T0

class Geometry:
    def __init__(self, xs, ys):
        """Class for representing the inner contour of a rocket engine, from the beginning of the combustion chamber to the nozzle exit.

        Args:
            xs (list): Array of x-positions, that the 'y' list corresponds to (m). Must be increasing values of x.
            ys (list): Array, containing the distances from the engine centreline to the inner wall (m).

        Attributes:
            xt (float): x-position of the throat (m)
            Rt (float): Throat radius (m)
            At (float): Throat area (m2)
            Re (float): Exit radius (m)
            Ae (float): Exit area (m2)
        """
        self.xs = xs
        self.ys = ys

    @property
    def xt(self):
        return self.xs[np.argmin(self.ys)]

    @property
    def Rt(self):
        return min(self.ys)

    @property
    def At(self):
        return np.pi * self.Rt**2

    @property
    def Re(self):
        return self.ys[-1]
    
    @property
    def Ae(self):
        return np.pi * self.Re**2


    def y(self, x):
        """Get the distance from the centreline to the inner wall of the engine.

        Args:
            x (float): x position (m)

        Returns:
            float: Distance from engine centreline to edge of inner wall (m)
        """
        return np.interp(x, self.xs, self.ys)

    def A(self, x):
        """Get the flow area for the exhaust gas

        Args:
            x (float): x position (m)

        Returns:
            float: Flow area (m2)
        """
        return np.pi * self.y(x)**2

class Wall:
    def __init__(self, material, thickness):
        """Object for representing an engine wall.

        Args:
            material (Material): Material object to define the material the wall is made of.
            thickness (float or callable): Thickness of the wall (m). Can be a constant float, or a function of position, i.e. t(x).
        """
        self.material = material
        self.thickness = thickness

        assert type(self.thickness) is float or type(self.thickness) is callable, "'thickness' input must be a float or callable"

class CoolingChannel:
    def __init__(self, type, T_c_in, p0_c_in, mdot_coolant, channel_height, coolant_transport):
        pass

class Engine:
    """Class for representing a liquid rocket engine.

    Args:
        gas (PerfectGas): Gas representing the exhaust gas for the engine.
        chamber_conditions (CombustionChamber): CombustionChamber for the engine.
        geometry (Geometry): Geomtry object to define the engine's contour.

    Keyword Args:
        engine_wall (Wall or list): Either a single Wall object that specifies the combustion chamber wall, or a list of Wall objects that represent multiple layers with different materials. First item in the list (index 0) touches the hot gas.
        cooling_jacket (CoolingJacket): CoolingJacket object to specify the cooling jacket on the engine.
        exhaust_transport (TransportProperties): TransportProperties object that defines the exhaust gas transport properties.

    Attributes:
        mdot (float): Mass flow rate of exhaust gas (kg/s)
        c_star (float): C* for the engine (m/s).

    """
    def __init__(self, perfect_gas, chamber_conditions, geometry, **kwargs):
        self.perfect_gas = perfect_gas
        self.chamber_conditions = chamber_conditions
        self.geometry = geometry

        # Find the choked mass flow rate
        self.mdot = bamboo.isen.get_choked_mdot(self.perfect_gas, self.chamber_conditions, self.geometry.At)

        # C* value, for convenience later
        self.c_star = self.chamber_conditions.p0 * self.geometry.At / self.mdot

        # Note that you can represent multiple layers of materials by giving a list as 'engine_wall'
        if "engine_wall" in kwargs:
            self.engine_wall = kwargs["engine_wall"]

            # Multiple layers of wall
            if type(self.engine_wall) is list:
                for item in self.engine_wall:
                    assert type(item) is Wall, "All items in the engine_wall list must be a Wall object. Otherwise a single Wall object must be given."
            
            # Only a single layer of wall
            else:
                assert type(self.engine_wall) is Wall, "You must give a Wall object as an input for engine_wall, or a list of engine_wall objects"

        if "cooling_jacket" in kwargs:
            self.cooling_jacket = kwargs["cooling_jacket"]
        
        if "exhaust_transport" in kwargs:
            self.exhaust_transport = kwargs["exhaust_transport"]  

    def M(self, x):
        """Get exhaust gas Mach number.

        Args:
            x (float): Axial position along the engine (m). 

        Returns:
            float: Mach number of the freestream.
        """
        #If we're at the throat then M = 1 by default:
        if x - self.geometry.xt <= 1e-12:
            return 1.00

        #If we're not at the throat:
        else:
            # Collect the relevant variables
            A = self.geometry.A(x)
            mdot = self.chamber_conditions.mdot
            p0 = self.chamber_conditions.p0
            cp = self.perfect_gas.cp
            T0 = self.chamber_conditions.T0
            gamma = self.perfect_gas.gamma

            # Function to find the root of
            def func_to_solve(Mach):
                return mdot * (cp * T0)**0.5 / (A  * p0) - bamboo.isen.m_bar(M = Mach, gamma = gamma)
            
            if x > self.geometry.xt:
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
        return self.isen.T(self.chamber_conditions.T0, self.M(x), self.perfect_gas.gamma)

    def p(self, x):
        """Get pressure at a position along the nozzle.
        Args:
            x (float): Distance from the throat, along the centreline (m)
        Returns:
            float: Pressure (Pa)
        """
        return self.isen.p(self.chamber_conditions.p0, self.M(x), self.perfect_gas.gamma)

    def rho(self, x):
        """Get exhaust gas density.
        Args:
            x (float): Axial position. Throat is at x = 0.
        Returns:
            float: Freestream gas density (kg/m3)
        """
    
        return self.p(x) / (self.T(x) * self.perfect_gas.R) # p = rho R T for an ideal gas, so rho = p/RT


    def steady_cooling_simulation(self):
        # Check that we have all the required inputs.
        assert hasattr(self, "cooling_jacket"), "'cooling_jacket' input must be given to Engine object in order to run a steady cooling simulation"
        assert hasattr(self, "exhaust_transport"), "'exhaust_transport' input must be given to Engine object in order to run a steady cooling simulation"
        assert hasattr(self, "engine_wall"), "'engine_wall' input must be given to Engine object in order to run a cooling simulation"

