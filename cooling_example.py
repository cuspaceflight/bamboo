import bamboo as bam
import bamboo.cooling as cool
import numpy as np
import matplotlib.pyplot as plt
import pypropep as ppp
import thermo
import time

'''Chamber conditions'''
Ac = 116.6e-4           #Chamber cross-sectional area (m^2)
pc = 15e5               #Chamber pressure (Pa)
mdot = 4.757            #Mass flow rate (kg/s)
p_amb = 1.01325e5       #Ambient pressure (Pa). 1.01325e5 is sea level atmospheric.
OF_ratio = 4            #Mass ratio

'''We want to investigate adding water to the isopropyl alcohol'''
water_mass_fraction = 0.10  #Fraction of the fuel that is water, by mass

'''Get combustion properties from pypropep'''
#Initialise and get propellants
ppp.init()
e = ppp.Equilibrium()
ipa = ppp.PROPELLANTS['ISOPROPYL ALCOHOL']
water = ppp.PROPELLANTS['WATER']
n2o = ppp.PROPELLANTS['NITROUS OXIDE']

#Add propellants by mass fractions (note the mass fractions can add up to more than 1)
e.add_propellants_by_mass([(ipa, 1-water_mass_fraction), 
                           (water, water_mass_fraction), 
                           (n2o, OF_ratio)])

#Adiabatic combustion using chamber pressure                      
e.set_state(P = pc/1e5, type='HP')                      

gamma = e.properties.Isex   #I don't know why they use 'Isex' for gamma. 
cp = 1000*e.properties.Cp   #Cp is given in kJ/kg/K, we want J/kg/K
Tc = e.properties.T

'''Get physical properties of the gas using thermo - use the mole fractions given by pypropep'''
#Exclude H2 and CO- it has very weird Prandtl number effects - it jumps around massively and I'm not sure why.
#Note that if you want carbon monoxide included you must type 'carbon monoxide', not 'CO' - the latter seems to make thermo use methanol (I don't know why)
thermo_gas = thermo.mixture.Mixture(['N2', 'H2O', 'CO2'], 
                                    zs = [e.composition['N2'], e.composition['H2O'], e.composition['CO2']])   

'''Engine dimensions'''
chamber_length = 75e-2
wall_thickness = 2e-3

'''Coolant jacket'''
OF_mass_ratio = 3
mdot_coolant = mdot/(OF_mass_ratio + 1)
channel_width = 1e-3
channel_height = 1e-3
k_wall = 300
inlet_T = 298.15    #Coolant inlet temperature
thermo_coolant = thermo.chemical.Chemical('Isopropyl Alcohol')

'''Create the engine object'''
perfect_gas = bam.PerfectGas(gamma = gamma, cp = cp)    #Gas for frozen flow
chamber = bam.ChamberConditions(pc, Tc, mdot)
nozzle = bam.Nozzle.from_engine_components(perfect_gas, chamber, p_amb, type = "rao", length_fraction = 0.8)
white_dwarf = bam.Engine(perfect_gas, chamber, nozzle)

print(f"Sea level thrust = {white_dwarf.thrust(1e5)/1000} kN")
print(f"Sea level Isp = {white_dwarf.isp(1e5)} s")

'''Cooling system setup'''
cooling_jacket = cool.CoolingJacket(k_wall, channel_width, channel_height, inlet_T, pc, thermo_coolant, mdot_coolant)
engine_geometry = cool.EngineGeometry(nozzle, chamber_length, Ac, wall_thickness)
cooled_engine = cool.EngineWithCooling(chamber, engine_geometry, cooling_jacket, perfect_gas, thermo_gas)

'''Plots'''
engine_geometry.plot_geometry()
#cooled_engine.show_gas_temperature()

'''Run the cooling system simulation'''
cooling_data = cooled_engine.run_heating_analysis(number_of_points = 1000, h_gas_model = "standard")

'''Plot the results'''
#Nozzle shape
shape_x = np.linspace(engine_geometry.x_min, engine_geometry.x_max, 1000)
shape_y = np.zeros(len(shape_x))

for i in range(len(shape_x)):
    shape_y[i] = engine_geometry.y(shape_x[i])

#Temperatures
fig, ax_T = plt.subplots()
ax_T.plot(cooling_data["x"], cooling_data["T_wall_inner"] - 273.15, label = "Wall (Inner)")
ax_T.plot(cooling_data["x"], cooling_data["T_wall_outer"]- 273.15, label = "Wall (Outer)")
ax_T.plot(cooling_data["x"], cooling_data["T_coolant"]- 273.15, label = "Coolant")
#ax_T.plot(cooling_data["x"], cooling_data["T_gas"], label = "Exhaust gas")
if cooling_data["boil_off_position"] != None:
    ax_T.axvline(cooling_data["boil_off_position"], color = 'red', linestyle = '--', label = "Coolant boil-off")

ax_T.grid()
ax_T.set_xlabel("Position (m)")
ax_T.set_ylabel("Temperature (deg C)")
ax_T.legend()

ax_shape = ax_T.twinx()
ax_shape.plot(shape_x, shape_y, color="blue", label = "Engine contour")
ax_shape.plot(shape_x, -shape_y, color="blue")
ax_shape.set_aspect('equal')
ax_shape.legend(loc = "lower left")

#Heat transfer coefficients
h_figs, h_axs = plt.subplots()
h_axs.plot(cooling_data["x"], cooling_data["h_gas"], label = "h_gas")
h_axs.plot(cooling_data["x"], cooling_data["h_coolant"], label = "h_coolant", )
if cooling_data["boil_off_position"] != None:
    h_axs.axvline(cooling_data["boil_off_position"], color = 'red', linestyle = '--', label = "Coolant boil-off")
h_axs.grid()
h_axs.legend()

plt.show()