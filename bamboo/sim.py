"""
Environment for simulating the thermal aspects of the engine.

Notation:
 - 'c': Cold side (usually coolant)
 - 'h': Hot side (usually exhaust gas)
 - 'w': At the wall (e.g. T_cw is the wall temperature on the cold side)
"""

from bamboo.circuit import ThermalCircuit

class CoolingSimulation:
    def __init__(self, T_c_in, T_h, p0_c_in, p_c, cp_c, mdot_c, R_th, dp_dx, x0, dx, x_end):

        self.T_c_in = T_c_in        # Constant                  - Coolant inlet temperature (K)
        self.T_h = T_h              # Function of 'state'       - Exhaust gas temperature (K)
        self.p0_c_in = p0_c_in      # Constant                  - Coolant inlet stagnation pressure (Pa)
        self.cp_c = cp_c            # Function of 'state''      - Coolant isobaric specific heat capacity (J/kg/K)
        self.mdot_c = mdot_c        # Constant                  - Coolant mass flow rate (kg/s)
        self.R_th = R_th            # Function of 'state'       - List of thermal resistances
        self.dp_dx = dp_dx          # Function of 'state'       - Stagnation pressure drop per unit length (Pa/m)

        self.x0 = x0                # Constant                  - Initial value of x to start at (m)
        self.dx = dx                # Constant                  - dx to move by for each step, corresponding to the direction that coolant flows in. Usually negative (if exhaust flows in positive x) (m)
        self.x_end = x_end          # Constant                  - Value of x to stop at (m)

        # Initialise our 'state' list
        self.i = 0
        self.state = [{}] * int( abs((x_end - x0) / dx) )       # Empty list of dictionaries
        self.state[self.i]["x"] = x0
        self.state[self.i]["T_c"] = self.T_c_in
        self.state[self.i]["T_cw"] = self.state["T_c"]
        self.state[self.i]["T_hw"] = self.T_h(self.state)
        self.state[self.i]["p0_c"] = self.p0_c_in

    def iterate(self):
        """
        Iterate one step at the current 'x' position. 
        """

        circuit = ThermalCircuit(T1 = self.T_h(self.state[self.i]), 
                                 T2 = self.state[self.i]["T_c"],
                                 R = self.R_th(self.state[self.i]) )

        self.state[self.i]["circuit"] = circuit
        self.state[self.i]["T_hw"] = circuit.T[1]
        self.state[self.i]["T_cw"] = circuit.T[-2]
        self.state[self.i]["p0_c"] = self.p0_c_in

    def step(self):
        """
        Move 'dx' onto the next x position, using the state from the previous position to calculate property changes.
        """

        old_state = self.state[self.i]

        self.i += 1
        self.state[self.i]["x"] = old_state["x"] + self.dx
        self.state[self.i]["T_c"] = old_state["T_c"] + old_state["circuit"].Qdot / (self.mdot_c * self.cp_c(old_state) ) # Temperature rise due to heat transfer in
        self.state[self.i]["T_cw"] = old_state["T_cw"]
        self.state[self.i]["T_hw"] = old_state["T_hw"]
        self.state[self.i]["p0_c"] = old_state["p0_c"] - self.dp_dx(old_state) * abs(self.dx)
