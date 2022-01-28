import numpy as np
import json

# Convert Curley data from .CSV to .JSON
raw = {}

output = {}

# Retrieve data from .CSV
raw["Contour"] = np.loadtxt("Pavli/Pavli_1966_Fig3.csv", delimiter =',', skiprows = 1)
raw["Channel width"] = np.loadtxt("Pavli/Pavli_1966_Fig22.csv", delimiter =',', skiprows = 1)
raw["Coolant pressure"] = np.loadtxt("Pavli/Pavli_1966_Table2a.csv", delimiter =',', skiprows = 1)
raw["Coolant temperature"] = np.loadtxt("Pavli/Pavli_1966_Table2b.csv", delimiter =',', skiprows = 1)

# Rearrange into a convenient format
output["Chamber Contour"] = {"x (m)" : raw["Contour"][:,0],
                             "y (m)" : raw["Contour"][:,1]}

output["Channel width"] = {"x (m)" : raw["Channel width"][:,0],
                           "w (m)" : raw["Channel width"][:,1]}

output["Coolant static pressure"] = {"x (m)" : raw["Coolant pressure"][:,1],
                                     "p (Pa)" : raw["Coolant pressure"][:,2]}

output["Coolant temperature"] = {"x (m)" : raw["Coolant temperature"][:,1],
                                 "T (K)" : raw["Coolant temperature"][:,2]}


# Save everything to a .json file
# Encoder to make numpy arrays .json serializable https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

with open('pavli.json', 'w') as f:
    json.dump(output, cls = NumpyEncoder, fp = f)