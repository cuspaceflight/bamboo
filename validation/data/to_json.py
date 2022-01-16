import numpy as np
import json

# Convert Vulcain data from .CSV to .JSON
raw = {}

output = {"Kirner" : {},
          "Ljungkrona" : {},
          "Nyden" : {}}

# Collect the Kirner_1993 data
raw["Kirner_Tcool_Sim"] = np.loadtxt("Kirner_1993_Fig15_Sim.csv", delimiter =',', skiprows = 1)
raw["Kirner_Tcool_Exp"] = np.loadtxt("Kirner_1993_Fig15_Exp.csv", delimiter =',', skiprows = 1)
raw["Extension_Contour"] = np.loadtxt("Kirner_1993_Fig15_Contour.csv", delimiter =',', skiprows = 1)
raw["Chamber_Contour"] = np.loadtxt("Kirner_1993_Fig14_Contour.csv", delimiter =',', skiprows = 1)

# Convert arrays into dictionaries
output["Kirner"]["Chamber Contour"] = {"x" : raw["Chamber_Contour"][:,0],
                                       "y" : raw["Chamber_Contour"][:,1]}

output["Kirner"]["Extension Contour"] = {"x" : raw["Extension_Contour"][:,0],
                                         "y" : raw["Extension_Contour"][:,1]}

output["Kirner"]["Extension Coolant Temperature (Sim)"] = {"x" : raw["Kirner_Tcool_Sim"][:,0],
                                                           "y" : raw["Kirner_Tcool_Sim"][:,1]}

output["Kirner"]["Extension Coolant Temperature (Exp)"] = {"x" : raw["Kirner_Tcool_Exp"][:,0],
                                                           "y" : raw["Kirner_Tcool_Exp"][:,1]}

# Shift the extension data so all x-axes use the same reference
output["Kirner"]["Extension Contour"]["x"] = output["Kirner"]["Extension Contour"]["x"] + output["Kirner"]["Chamber Contour"]["x"][-1]
output["Kirner"]["Extension Coolant Temperature (Sim)"]["x"] = output["Kirner"]["Extension Coolant Temperature (Sim)"]["x"] + output["Kirner"]["Chamber Contour"]["x"][-1]
output["Kirner"]["Extension Coolant Temperature (Exp)"]["x"] = output["Kirner"]["Extension Coolant Temperature (Exp)"]["x"] + output["Kirner"]["Chamber Contour"]["x"][-1]

# Make an overall engine contour dataset
output["Kirner"]["Engine Contour"] = {"x" : list(output["Kirner"]["Chamber Contour"]["x"]) + list(output["Kirner"]["Extension Contour"]["x"]),
                                      "y" : list(output["Kirner"]["Chamber Contour"]["y"]) + list(output["Kirner"]["Extension Contour"]["y"])}

# Collect the Ljungkrona_1991 data
raw["Ljungkrona_T_Exp"] = np.loadtxt("Ljungkrona_1998_Fig8_Exp.csv", delimiter =',', skiprows = 1)
raw["Ljungkrona_T_Sim"] = np.loadtxt("Ljungkrona_1998_Fig8_Sim.csv", delimiter =',', skiprows = 1)
raw["Ljungkrona_h_Exp"]= np.loadtxt("Ljungkrona_1998_Fig9_Exp.csv", delimiter =',', skiprows = 1)
raw["Ljungkrona_h_Sim"]= np.loadtxt("Ljungkrona_1998_Fig9_Sim.csv", delimiter =',', skiprows = 1)

# Convert into dictionaries
output["Ljungkrona"]["Extension Coolant Temperature (Exp)"] = {"x" : raw["Ljungkrona_T_Exp"][:, 0],
                                                               "y" : raw["Ljungkrona_T_Exp"][:, 1]}

output["Ljungkrona"]["Extension Firewall Temperature (Exp)"] = {"x" : raw["Ljungkrona_T_Exp"][:-1, 2],
                                                                "y" : raw["Ljungkrona_T_Exp"][:-1, 3]}

output["Ljungkrona"]["Extension Firewall Temperature (Sim)"] = {"x" : raw["Ljungkrona_T_Sim"][:, 0],
                                                                "y" : raw["Ljungkrona_T_Sim"][:, 1]}

output["Ljungkrona"]["Extension Normalised Convective Coefficient (Exp)"] = {"x" : raw["Ljungkrona_h_Exp"][:, 0],
                                                                             "y" : raw["Ljungkrona_h_Exp"][:, 1]}

output["Ljungkrona"]["Extension Normalised Convective Coefficient (Sim)"] = {"x" : raw["Ljungkrona_h_Sim"][:, 0],
                                                                             "y" : raw["Ljungkrona_h_Sim"][:, 1]}

# Scale the x-axes and shift them so they match with the Kirner data

# WE CANNOT SIMPLY LINEARLY SCALE! THE NYDEN AND LJUNGKRONA DATA IS ALONG THE TUBE AXIS - WHICH SPIRALS OUTWARDS AS THE NOZZLE INCREASES IN DIAMETER
# WE MUST FIRST SET UP TWO ARRAYS TO CONVERT BETWEEN THE ENGINE AXIS (dx) AND THE FLOW PATH ALONG THE TUBE (dL)

x_positions = np.linspace(output["Kirner"]["Extension Contour"]["x"][0], output["Kirner"]["Extension Contour"]["x"][-1], 1000)  # Length along the x-axis
L_positions = np.zeros(len(x_positions))                                                                                        # Distance travelled along a spiralling cooling channel

wall_thickness = 0.4e-3
pitch = 1.824

for i in range(len(x_positions)):
    if i == 0:
        pass
    else:
        x = x_positions[i]
        y = np.interp(x, output["Kirner"]["Extension Contour"]["x"], output["Kirner"]["Extension Contour"]["y"])
        R_inner = y + wall_thickness
        spiral_angle = np.arctan(2 * np.pi * R_inner / pitch)
        dL_dx = 1 / np.cos(spiral_angle)                            # dL/dx, i.e. length travelled along the spiral channel for each 'dx'
        dx = x_positions[i] - x_positions[i-1]

        L_positions[i] = L_positions[i-1] + dx * dL_dx
                
                               

for key in output["Ljungkrona"].keys():
    output["Ljungkrona"][key]["x"] = np.interp(x = output["Ljungkrona"][key]["x"], 
                                               xp = L_positions,
                                               fp = x_positions)

# Collect the Nyden_1991 data
raw["Nyden_Twc_Exp"] = np.loadtxt("Nyden_1991_Fig2_Twc_Exp.csv", delimiter =',', skiprows = 1)
raw["Nyden_Twc_Sim"] = np.loadtxt("Nyden_1991_Fig2_Twc_Sim.csv", delimiter =',', skiprows = 1)

# I'm not sure if the 'experimental' data is actually experimental
output["Nyden"]["Cooling Side Wall Temperature (Sim)"] = {"x" : raw["Nyden_Twc_Sim"][:, 0],
                                                          "y" : raw["Nyden_Twc_Sim"][:, 1]}

# Scale the x-axes and shift them so they match with the Kirner data
for key in output["Nyden"].keys():
    output["Nyden"][key]["x"] = np.interp(x = output["Nyden"][key]["x"], 
                                               xp = L_positions,
                                               fp = x_positions)


# Save everything to a .json file

# Encoder to make numpy arrays .json serializable https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

with open('vulcain.json', 'w') as f:
    json.dump(output, cls = NumpyEncoder, fp = f)