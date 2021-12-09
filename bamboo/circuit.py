"""
Class to simplify thermal circuit calculations
"""

import numpy as np

class ThermalCircuit:
    def __init__(self, T1, T2, R):
        """Class for solving thermal circuits. Will solve them upon initialising.

        Args:
            T1 (float): Temperature at start
            T2 (float): Temperature at end
            R (list): List of resistances between T1 and T2

        Attributes:
            Qdot (float): Heat transfer rate (positive in the direction of T1 --> T2)
            T (list): List of temperatures in between each resistance, including T1 and T2 at either end. i.e. [T1, ..., T2].
        """
        self.R = R
        self.T1 = T1
        self.T2 = T2

        self.Qdot = (T1 - T2)/sum(R)
        self.T = np.zeros(len(R) + 1)
        self.T[0] = T1

        for i in range(1, len(R)):
            self.T[i] = self.T[i-1] - self.Qdot*R[i-1]
