import numpy as np
import json

# Convert Vulcain data from .CSV to .JSON
raw = {}

output = {"Kirner" : {},
          "Ljungkrona" : {},
          "Nyden" : {}}

# Collect the Kirner_1993 data
raw["Kirner_Tcool_Sim"] = np.loadtxt("Vulcain/Kirner_1993_Fig15_Sim.csv", delimiter =',', skiprows = 1)
raw["Kirner_Tcool_Exp"] = np.loadtxt("Vulcain/Kirner_1993_Fig15_Exp.csv", delimiter =',', skiprows = 1)

