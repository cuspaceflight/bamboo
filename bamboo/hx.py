"""
General solver for coflow and counterflow heat exchangers, using a 1-D thermal resistance model.

Notation:
 - 'c': Cold side (usually coolant)
 - 'h': Hot side (usually exhaust gas)
 - 'w': At the wall (e.g. T_cw is the wall temperature on the cold side)
"""

from bamboo.circuit import ThermalCircuit

class HXSolver:
    def __init__(self, T_c_in, T_h, p0_c_in, cp_c, mdot_c, R_th, extra_dQ_dx, dp_dx, x_start, dx, x_end):
        """Class for solving heat exchanger problems.

        Args:
            T_c_in (float): Coolant inlet temperature (K)
            T_h (callable): Exhaust gas temperature (K). Must be a function of 'state'.
            p0_c_in (float): Coolant inlet stagnation pressure (Pa)
            cp_c (callable): Coolant isobaric specific heat capacity (J/kg/K). Must be a function of 'state'.
            mdot_c (float): Coolant mass flow rate (kg/s)
            R_th (callable): List of thermal resistances [R1, R2 ... etc], in the order T_cold --> T_hot. Note they need to be 1D resistances, so Qdot is per unit length. Must be a function of 'state'.
            extra_dQ_dx (callable): Extra heat transfer rate (positive into the coolant), to add on (W), to represent things like fins protruding into the coolant flow. Must be a function of 'state'.
            dp_dx (callable): Stagnation pressure drop per unit length (Pa/m)
            x_start (float): Initial value of x to start at (m)
            dx (float): dx to move by for each step, corresponding to the direction that coolant flows in. Usually negative for counterflow heat exchanger (m)
            x_end (float): Value of x to stop at (m)
        """

        self.T_c_in = T_c_in       
        self.T_h = T_h             
        self.p0_c_in = p0_c_in      
        self.cp_c = cp_c          
        self.mdot_c = mdot_c      
        self.R_th = R_th            
        self.extra_dQ_dx = extra_dQ_dx
        self.dp_dx = dp_dx         
        self.x_start = x_start     
        self.dx = dx                
        self.x_end = x_end         

        self.reset()
    
    def reset(self):
        """
        Reset our 'state' list to the initial conditions and set self.i to zero.
        """
        self.i = 0
        
        # Set up an empty list of dictionaries
        self.state = [None] * int( abs((self.x_end - self.x_start) / self.dx) )    
        for i in range(len(self.state)):
            self.state[i] = {}

        self.state[self.i]["x"] = self.x_start
        self.state[self.i]["T_c"] = self.T_c_in
        self.state[self.i]["T_cw"] = self.state[self.i]["T_c"]
        self.state[self.i]["T_hw"] = self.T_h(self.state[self.i])
        self.state[self.i]["p0_c"] = self.p0_c_in

    def iterate(self):
        """
        Iterate one step at the current 'x' position. 
        """

        circuit = ThermalCircuit(T1 = self.state[self.i]["T_c"], 
                                 T2 = self.T_h(self.state[self.i]),
                                 R = self.R_th(self.state[self.i]) )       

        self.state[self.i]["circuit"] = circuit
        self.state[self.i]["T_hw"] = circuit.T[-2]
        self.state[self.i]["T_cw"] = circuit.T[1]

    def step(self):
        """
        Move 'dx' onto the next x position, using the state from the previous position to calculate property changes.
        """

        old_state = self.state[self.i]

        self.i += 1
        self.state[self.i]["x"] = old_state["x"] + self.dx

        Q_tot = old_state["circuit"].Qdot - self.extra_dQ_dx(old_state) * abs(self.dx)     # extra_Q is positive into the coolant, but circuit.Qdot is positive into the exhaust
        
        self.state[self.i]["T_c"] = old_state["T_c"] - Q_tot * abs(self.dx) / (self.mdot_c * self.cp_c(old_state) ) # Temperature rise due to heat transfer in - note the Qdot is per unit length
        self.state[self.i]["T_cw"] = old_state["T_cw"]
        self.state[self.i]["T_hw"] = old_state["T_hw"]
        self.state[self.i]["p0_c"] = old_state["p0_c"] - self.dp_dx(old_state) * abs(self.dx)

    def run(self, iter_start = 5, iter_each = 1):
        """Run the simulation until we reach x >= x_end.

        Args:
            iter_start (int, optional): Number of iterations to use on the first gridpoint. Defaults to 5.
            iter_each (int, optional): Number of iterations to use on each intermediate grid point. Defaults to 1.
        """
        assert type(iter_start) is int, "'iter_start' must be an integer"
        assert iter_start >= 1, "'iter_start' must be at least 1"

        assert type(iter_each) is int, "'iter_each' must be an integer"
        assert iter_each >= 1, "'iter_each' must be at least 1"

        # Initialise our 'state'
        self.reset()

        # Perform the required amount of iterations on the first grid point
        counter = 0
        while counter < iter_start:
            self.iterate()
            counter += 1

        while self.i < len(self.state) - 1:
            # Move to next grid point
            self.step()

            # Perform the required number of iterations
            counter = 0
            while counter < iter_each:
                self.iterate()
                counter += 1
