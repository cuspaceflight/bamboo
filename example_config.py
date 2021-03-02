import bamboo as bam
import bamboo.cooling as cool
import pypropep as ppp
import thermo
import numpy as np

'''Simulation properties'''
points = 4000 # Must be consistent to use liner optimisation

'''Chamber conditions'''
Ac = 116.6e-4  # Chamber cross-sectional area (m^2)
pc = 15e5  # Chamber pressure (Pa)
mdot = 4.757  # Mass flow rate (kg/s)
p_amb = 1.01325e5  # Ambient pressure (Pa). 1.01325e5 is sea level atmospheric.
OF_ratio = 4  # Mass ratio

'''Wall material properties - midranges values taken for pure copper, granta edupack 2020 '''  # What would I do without 1A materials
wall_modulus = 130E9  # Young's modulus (Pa)
wall_yield = 195E6  # Yield stress (Pa)
wall_poisson = 0.345  # Poisson's ratio
wall_expansion = 17.35E-6  # Thermal expansion coefficient (strain/K)
wall_conductivity = 292  # Thermal conductivity (W/mK)

'''We want to investigate adding water to the isopropyl alcohol'''
water_mass_fraction = 0.10  # Fraction of the fuel that is water, by mass

'''Get combustion properties from pypropep'''
# Initialise and get propellants
ppp.init()
e = ppp.Equilibrium()
ipa = ppp.PROPELLANTS['ISOPROPYL ALCOHOL']
water = ppp.PROPELLANTS['WATER']
n2o = ppp.PROPELLANTS['NITROUS OXIDE']

# Add propellants by mass fractions (note the mass fractions can add up to
# more than 1)
e.add_propellants_by_mass([(ipa, 1 - water_mass_fraction),
                           (water, water_mass_fraction),
                           (n2o, OF_ratio)])

# Adiabatic combustion using chamber pressure
e.set_state(P=pc / 1e5, type='HP')

gamma = e.properties.Isex  # I don't know why they use 'Isex' for gamma.
cp = 1000 * e.properties.Cp  # Cp is given in kJ/kg/K, we want J/kg/K
Tc = e.properties.T

'''Get physical properties of the gas using thermo - use the mole fractions given by pypropep'''
# Exclude H2 and CO- it has very weird Prandtl number effects - it jumps around massively and I'm not sure why.
# Note that if you want carbon monoxide included you must type 'carbon
# monoxide', not 'CO' - the latter seems to make thermo use methanol (I
# don't know why)
thermo_gas = thermo.mixture.Mixture(['N2', 'H2O', 'CO2'], zs=[
                                    e.composition['N2'], e.composition['H2O'], e.composition['CO2']])

'''Engine dimensions'''
chamber_length = 0.170  # 75e-2
wall_thickness = np.array(points * [3e-3])

'''Coolant jacket'''
OF_mass_ratio = 3
mdot_coolant = mdot / (OF_mass_ratio + 1)
semi_circle_diameter = 4e-3
inlet_T = 298.15  # Coolant inlet temperature
thermo_coolant = thermo.chemical.Chemical('Isopropyl Alcohol')

'''Create the engine object'''
perfect_gas = bam.PerfectGas(gamma=gamma, cp=cp)  # Gas for frozen flow
chamber = bam.ChamberConditions(pc, Tc, mdot)
nozzle = bam.Nozzle.from_engine_components(
    perfect_gas, chamber, p_amb, type="rao", length_fraction=0.8)
white_dwarf = bam.Engine(perfect_gas, chamber, nozzle)

print(f"Sea level thrust = {white_dwarf.thrust(p_amb)/1000} kN")
print(f"Sea level Isp = {white_dwarf.isp(p_amb)} s")

'''Cooling system setup'''
wall_material = cool.Material(
    wall_modulus,
    wall_yield,
    wall_poisson,
    wall_expansion,
    wall_conductivity)
cooling_jacket = cool.CoolingJacket(
    wall_material,
    inlet_T,
    pc,
    thermo_coolant,
    mdot_coolant,
    channel_shape="semi-circle",
    circle_diameter=semi_circle_diameter)
engine_geometry = cool.EngineGeometry(
    nozzle, chamber_length, Ac, wall_thickness)
cooled_engine = cool.EngineWithCooling(
    chamber,
    engine_geometry,
    cooling_jacket,
    perfect_gas,
    thermo_gas)