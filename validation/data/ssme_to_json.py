import numpy as np
import json

# Convert Curley data from .CSV to .JSON
raw = {}

output = {}

# Retrieve data from .CSV
raw["Contour"] = np.loadtxt("SSME/Pizzarelli_Fig4_r.csv", delimiter =',', skiprows = 1)
raw["Channel height"] = np.loadtxt("SSME/Pizzarelli_Fig4_h.csv", delimiter =',', skiprows = 1)
raw["Channel width"] = np.loadtxt("SSME/Pizzarelli_Fig4_b.csv", delimiter =',', skiprows = 1)
raw["Wall thickness"] = np.loadtxt("SSME/Pizzarelli_Fig4_sw.csv", delimiter =',', skiprows = 1)
raw["Rib thickness"] = np.loadtxt("SSME/Pizzarelli_Fig4_tw.csv", delimiter =',', skiprows = 1)
raw["Coolant temperature"] = np.loadtxt("SSME/Pizzarelli_Fig7b.csv", delimiter =',', skiprows = 1)

# Rearrange into a convenient format
output["Chamber contour"] = {"x (m)" : raw["Contour"][:,0],
                             "y (m)" : raw["Contour"][:,1]}

output["Channel height"] = {"x (m)" : raw["Channel height"][:,0],
                            "h (m)" : raw["Channel height"][:,1]}


output["Channel width"] = {"x (m)" : raw["Channel width"][:,0],
                           "w (m)" : raw["Channel width"][:,1]}


output["Wall thickness"] = {"x (m)" : raw["Wall thickness"][:,0],
                            "t (m)" : raw["Wall thickness"][:,1]}

output["Rib thickness"] = {"x (m)" : raw["Rib thickness"][:,0],
                           "t (m)" : raw["Rib thickness"][:,1]}

output["Coolant temperature"] = {"x (m)" : raw["Coolant temperature"][:,0],
                                 "T (K)" : raw["Coolant temperature"][:,1]}


# Save everything to a .json file
# Encoder to make numpy arrays .json serializable https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

with open('ssme.json', 'w') as f:
    json.dump(output, cls = NumpyEncoder, fp = f)