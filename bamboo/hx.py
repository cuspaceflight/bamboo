"""
General solver for coflow and counterflow heat exchangers, using a 1-D thermal resistance model.

Notation:
 - 'c': Cold side (usually coolant)
 - 'h': Hot side (usually exhaust gas)
 - 'w': At the wall (e.g. T_cw is the wall temperature on the cold side)
"""

from bamboo.circuit import ThermalCircuit

class HXSolver:
    def __init__(self, T_c_in, T_h, p_c_in, cp_c, mdot_c, V_c, A_c, Rdx, extra_dQ_dx, dp_dx_f, x_start, dx, x_end):
        """Class for solving heat exchanger problems.

        Args:
            T_c_in (float): Coolant inlet static temperature (K)
            T_h (callable): Exhaust gas static temperature (K). Must be a function of 'state'.
            p_c_in (float): Coolant inlet static pressure (Pa)
            cp_c (callable): Coolant isobaric specific heat capacity (J/kg/K). Must be a function of 'state'.
            mdot_c (float): Coolant mass flow rate (kg/s)
            V_c (callable): Coolant velocity (m/s). Must be a function of 'state'.
            A_c (callable): Coolant flow area (m2). Must be a function of 'state'.
            Rdx (callable): List of thermal resistances [R1, R2 ... etc], in the order T_cold --> T_hot. Note they need to be 1D resistances, so Qdot is per unit length. Must be a function of 'state'.
            extra_dQ_dx (callable): Extra heat transfer rate (positive into the coolant), to add on (W), to represent things like fins protruding into the coolant flow. Must be a function of 'state'.
            dp_dx_f (callable): Frictional pressure drop per unit length (Pa/m)
            x_start (float): Initial value of x to start at (m)
            dx (float): dx to move by for each step, corresponding to the direction that coolant flows in. Usually negative for counterflow heat exchanger (m)
            x_end (float): Value of x to stop at (m)
        """

        self.T_c_in = T_c_in     
        self.T_h = T_h             
        self.p_c_in = p_c_in      
        self.cp_c = cp_c          
        self.mdot_c = mdot_c  
        self.V_c = V_c    
        self.A_c = A_c
        self.Rdx = Rdx            
        self.extra_dQ_dx = extra_dQ_dx
        self.dp_dx_f = dp_dx_f         
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

        self.state[0]["x"] = self.x_start
        self.state[0]["p_c"] = self.p_c_in
        self.state[0]["T_c"] = self.T_c_in
        self.state[0]["T_cw"] = self.state[0]["T_c"]
        self.state[0]["T_hw"] = self.T_h(self.state[0])
        self.state[0]["V_c"] = self.V_c(self.state[0])
        self.state[0]["cp_c"] = self.cp_c(self.state[0])

        # Initial guess for the next T_c, T_wc, T_wh, and p_c
        self.state[1]["x"] = self.state[0]["x"] + self.dx
        self.state[1]["T_c"] = self.state[0]["T_c"]
        self.state[1]["T_cw"] = self.state[0]["T_cw"] 
        self.state[1]["T_hw"] = self.state[0]["T_hw"]
        self.state[1]["p_c"] = self.state[0]["p_c"] 

    def iterate(self):
        """
        Iterate one step at the current 'x' position. 
        """
        i = self.i 
        #print(f'{100*abs((self.state[i]["x"] - self.x_start) / (self.x_start - self.x_end)):.2f}%, Tc = {self.state[i]["T_c"]}, pc = {self.state[i]["p_c"]}')

        # Calculate thermal resistance and solve thermal circuit
        self.state[i]["circuit"] = ThermalCircuit(T1 = self.state[self.i]["T_c"], 
                                                  T2 = self.T_h(self.state[i]),
                                                  R = self.Rdx(self.state[i]))       

        self.state[i]["T_hw"] = self.state[i]["circuit"].T[-2]
        self.state[i]["T_cw"] = self.state[i]["circuit"].T[1]

        # For the last point we only need to iterate for wall temperature
        if i != len(self.state) - 1:
            dQ_dx_i = - self.state[i]["circuit"].Qdot #+ self.extra_dQ_dx(self.state[i])       # extra_Q is positive into the coolant, but circuit.Qdot is positive into the exhaust

            # Steady flow energy equation to get the i+1 coolant temperature
            self.state[i]["cp_c"] = self.cp_c(self.state[i])
            self.state[i+1]["cp_c"] = self.cp_c(self.state[i+1])
            cp_mean = (self.state[i]["cp_c"] + self.state[i+1]["cp_c"]) / 2

            self.state[i]["V_c"] = self.V_c(self.state[i])
            self.state[i+1]["V_c"] = self.V_c(self.state[i+1])

            self.state[i+1]["T_c"] = self.state[i]["T_c"]                                                               \
                                    + 0.5 * (self.state[i]["V_c"]**2 / cp_mean - self.state[i+1]["V_c"]**2 / cp_mean)   \
                                    + 1.0 / (self.mdot_c * cp_mean) * dQ_dx_i * abs(self.dx)      

            # Momentum equation to get pressure drop
            self.state[i+1]["V_c"] = self.V_c(self.state[i+1])     # Update V_c[i+1], since we have a new T_c[i+1]

            dp_dx_f_i = self.dp_dx_f(self.state[i]) 

            self.state[i+1]["p_c"] = self.state[i]["p_c"] - self.mdot_c / self.A_c(self.state[i]) * (self.state[i+1]["V_c"] - self.state[i]["V_c"]) + dp_dx_f_i * self.dx

    def step(self):
        """
        Move 'dx' onto the next x position. Make an initial guess for T_c[i+2] and p_c[i+2] based on T_c[i+1] and p_c[i+1].
        """
        self.i += 1
        i = self.i
        #print(f"i = {i}")

        # Don't try and guess the future state if we're on the last grid point
        if i != len(self.state) - 1:           
            self.state[i+1]["x"] = self.state[i]["x"] + self.dx

            # Initial guess for the next T_c, T_wc, T_wh, and p_c
            self.state[i+1]["T_c"] = self.state[i]["T_c"] + self.dx
            self.state[i+1]["T_cw"] = self.state[i]["T_cw"] 
            self.state[i+1]["T_hw"] = self.state[i]["T_hw"] 
            self.state[i+1]["p_c"] = self.state[i]["p_c"] 


    def run(self, iter_start = 5, iter_each = 2):
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
