"""
General solver for coflow and counterflow heat exchangers, using a 1-D thermal resistance model.

Notation:
 - 'c': Cold side (usually coolant)
 - 'h': Hot side (usually exhaust gas)
 - 'w': At the wall (e.g. T_cw is the wall temperature on the cold side)
"""

from bamboo.circuit import ThermalCircuit

class HXSolver:
    def __init__(self, T_c_in, T_h, p0_c_in, cp_c, mdot_c, R_th, dp_dx, x_start, dx, x_end):

        self.T_c_in = T_c_in        # Constant                  - Coolant inlet temperature (K)
        self.T_h = T_h              # Function of 'state'       - Exhaust gas temperature (K)
        self.p0_c_in = p0_c_in      # Constant                  - Coolant inlet stagnation pressure (Pa)
        self.cp_c = cp_c            # Function of 'state''      - Coolant isobaric specific heat capacity (J/kg/K)
        self.mdot_c = mdot_c        # Constant                  - Coolant mass flow rate (kg/s)
        self.R_th = R_th            # Function of 'state'       - List of thermal resistances [R1, R2 ... etc], in the order T_cold --> T_hot. Note they need to be 1D resistances, so Qdot is per unit length
        self.dp_dx = dp_dx          # Function of 'state'       - Stagnation pressure drop per unit length (Pa/m)

        self.x_start = x_start      # Constant                  - Initial value of x to start at (m)
        self.dx = dx                # Constant                  - dx to move by for each step, corresponding to the direction that coolant flows in. Usually negative (if exhaust flows in positive x) (m)
        self.x_end = x_end          # Constant                  - Value of x to stop at (m)

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
        self.state[self.i]["T_c"] = old_state["T_c"] - old_state["circuit"].Qdot * abs(self.dx) / (self.mdot_c * self.cp_c(old_state) ) # Temperature rise due to heat transfer in - not the Qdot is per unit length
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

        #print(f"bamboo.sim.py: Simulation initialised, T_hw = {self.state[self.i]['T_hw']} and T_cw = {self.state[self.i]['T_cw']}")

        while self.i < len(self.state) - 1:
            # Move to next grid point
            self.step()

            # Perform the required number of iterations
            counter = 0
            while counter < iter_each:
                self.iterate()
                counter += 1

            #print(f"bamboo.sim.py: i = {self.i}, Tc = {self.state[self.i]['T_c']}, T_hw = {self.state[self.i]['T_hw']}, T_cw = {self.state[self.i]['T_cw']}, p0_c = {self.state[self.i]['p0_c']}")

