"""
Classes for representing solid materials, fluids and transport properties. Also contains some useful pre-defined materials.

References
- [1] - CoolProp, http://coolprop.org/
"""

# Classes
class Material:
    """Class used to specify a material and its properties. For calculating temperatures, only 'k' must be defined. For stresses, you also need E, alpha, and poisson.

    Args:
        k (float): Thermal conductivity (W/m/K)
        
    Keyword Args:
        E (float): Young's modulus (Pa)
        alpha (float): Thermal expansion coefficient (strain/K)
        poisson (float): Poisson's ratio
    """
    def __init__(self, k, **kwargs):
        self.k = k                  

        if "E" in kwargs:
            self.E = kwargs["E"]
        else:
            self.E = float("NaN")

        if "alpha" in kwargs:
            self.alpha = kwargs["alpha"]
        else:
            self.alpha = float("NaN")

        if "poisson" in kwargs:
            self.poisson = kwargs["poisson"]
        else:
            self.poisson = float("NaN")

class TransportProperties:
    def __init__(self, Pr, mu, k, cp = None, rho = None):
        """
        Container for specifying your transport properties. Each input can either be a function of temperature and pressure (in that order), e.g. mu(T, p). Otherwise they can be constant floats.

        Args:
            Pr (float or callable): Prandtl number.
            mu (float or callable): Absolute viscosity (Pa s).
            k (float or callable): Thermal conductivity (W/m/K).
            cp (float or callable, optional): Isobaric specific heat capacity (J/kg/K) - only required for coolants.
            rho (float or callable, optional): Density (kg/m^3) - only required for coolants.
        """

        self.type = type
        self._Pr = Pr
        self._mu = mu
        self._k = k
        self._rho = rho
        self._cp = cp

    def Pr(self, T, p):
        """Prandtl number.

        Args:
            T (float): Temperature (K)
            p (float): Pressure (Pa)

        Returns:
            float: Prandtl number
        """
        if type(self._Pr) is callable:
            return self._Pr(T, p)
        
        else:
            return self._Pr

    def mu(self, T, p):
        """Absolute viscosity (Pa s)

        Args:
            T (float): Temperature (K)
            p (float): Pressure (Pa)

        Returns:
            float: Absolute viscosity (Pa s)
        """
        if type(self._mu) is callable:
            return self._mu(T, p)
        
        else:
            return self._mu

    def k(self, T, p):
        """Thermal conductivity (W/m/K)

        Args:
            T (float): Temperature (K)
            p (float): Pressure (Pa)

        Returns:
            float: Thermal conductivity (W/m/K)
        """
        if type(self._k) is callable:
            return self._k(T, p)
        
        else:
            return self._k

    def rho(self, T, p):
        """Density (kg/m^3)
        Args:
            T (float): Temperature (K)
            p (float): Pressure (Pa)
        Returns:
            float: Density (kg/m^3)
        """
        if self._rho is None:
            raise ValueError("TransportProperties object does not have its density 'rho' defined. If you tried to use this TransportProperties object for a coolant, you need to specify the 'rho' input.")

        if type(self._rho) is callable:
            return self._rho(T, p)
        
        else:
            return self._rho

    def cp(self, T, p):
        """Isobaric specific heat capacity (J/kg/K)

        Args:
            T (float): Temperature (K)
            p (float): Pressure (Pa)

        Returns:
            float: Isobaric specific heat capacity (J/kg/K)
        """

        if self._cp is None:
            raise ValueError("TransportProperties object does not have its isobaric specific heat capacity 'cp' defined. If you tried to use this TransportProperties object for a coolant, you need to specify the 'cp' input.")

        if type(self._cp) is callable:
            return self._cp(T, p)
        
        else:
            return self._cp

# Solids
CopperC106 = Material(E = 117e9, poisson = 0.34, alpha = 16.9e-6, k = 391.2)
StainlessSteel304 = Material(E = 193e9, poisson = 0.29, alpha = 16e-6, k = 14.0)
Graphite = Material(E = float('NaN'), poisson = float('NaN'), alpha = float('NaN'), k = 63.81001)

# Fluids
Water = TransportProperties(Pr = 6.159, mu = 0.89307e-3, k = 0.60627, cp = 4181.38, rho =  997.085)         # Water at 298 K and 1 bar [1]
Ethanol = TransportProperties(Pr = 16.152, mu = 1.0855e-3, k = 0.163526, cp = 2433.31, rho = 785.26)        # Ethanol at 298 K and 1 bar [1]
CO2 = TransportProperties(mu = 3.74e-5, k =  0.0737, Pr = 0.72)                                             # Representative values for CO2 gas