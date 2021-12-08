import numpy as np
import matplotlib.pyplot as plt
import bamboo.rao
import bamboo.isen

R_BAR = 8.3144621e3         #Universal gas constant (J/K/kmol)

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
            raise ValueError(f"Not enough inputs provided to fully define the Gas. You must provide exactly 2, but you provided {len(kwargs)}.")

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

class Nozzle:
    """Object for calculating and storing nozzle geometry.

    Args:
        type (str, optional): Desired shape, can be "rao", "cone" or "custom". Defaults to "rao".

    Keyword Args:
        At (float): Throat area (m^2) - required for 'rao' and 'cone' nozzles.
        Ae (float): Exit plane area (m^2) - required for 'rao' and 'cone' nozzles.
        length_fraction (float, optional): Length fraction for 'rao' nozzle - used if type = "rao". Defaults to 0.8.
        theta_n (float, optional): theta_n for 'rao' nozzle - used if type = "rao". Defaults to interpolation ofthe graph in reference [1]
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
            if (self.Ae/self.At < 3.7 or self.Ae/self.At > 47) and "theta_n" not in kwargs:
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
                if "theta_n" in kwargs:
                    self.theta_n = kwargs["theta_n"]
                    self.theta_e_graph = float("NaN")
                else:
                    self.theta_n = bamboo.rao.rao_theta_n(self.Ae/self.At)             #Inflection angle (rad), as defined in [1]
                    self.theta_e_graph = bamboo.rao.rao_theta_e(self.Ae/self.At)       #Exit angle (rad) read off the Rao graph, as defined in [1] - not actually used in bamboo for the quadratic fitting (just here for reference)

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
    def from_engine_components(perfect_gas, chamber_conditions, p_e, mdot, type = "rao", **kwargs):
        """Generate nozzle based on given gas properties and combustion chamber conditions.

        Args:
            perfect_gas (PerfectGas): PerfectGas object for the exhaust gases.
            chamber_conditions (CombustionChamber): CombustionChamber object.
            p_e (float): Ambient pressure (Pa). The nozzle will be designed to have this pressure at exit.
            mdot (float): Mass flow rate of exhaust gas (kg/s).
            type (str, optional): Nozzle type. Can be "rao" or "cone". Defaults to "rao".
            
        Keyword Args:
            length_fraction (float, optional): Rao nozzle length fraction, as defined in Reference [1]. Only used if type = "rao". Defaults to 0.8.
            cone_angle (float, optional): Cone angle for a conical nozzle (deg). Only used if type = "cone". Defaults to 15 degrees.

        Returns:
            [Nozzle]: The nozzle object.
        """
        return Nozzle(At = bamboo.isen.get_throat_area(perfect_gas, chamber_conditions, mdot = mdot), 
                      Ae = bamboo.isen.get_exit_area(perfect_gas, chamber_conditions, p_e = p_e, mdot = mdot), 
                      type = type, 
                      **kwargs)



class Engine:
    """Class for representing a liquid rocket engine.

    Args:
        gas (PerfectGas): Gas representing the exhaust gas for the engine.
        chamber_conditions (CombustionChamber): CombustionChamber for the engine.
        nozzle (Nozzle): Nozzle for the engine.

    Attributes:
        mdot (float): Mass flow rate of exhaust gas (kg/s)
        c_star (float): C* for the engine (m/s).
        geometry (EngineGeometry): EngineGeometry object (if added).
    """
    def __init__(self, perfect_gas, chamber_conditions, nozzle):
        self.perfect_gas = perfect_gas
        self.chamber_conditions = chamber_conditions
        self.nozzle = nozzle

        # Find the choked mass flow rate
        self.mdot = bamboo.isen.get_choked_mdot(self.perfect_gas, self.chamber_conditions, self.nozzle.At)

        # Extra attributes
        self.c_star = self.chamber_conditions.p0 * self.nozzle.At / self.mdot
        self.has_exhaust_transport = False
        self.has_cooling_jacket = False
        self.has_insulator = False
