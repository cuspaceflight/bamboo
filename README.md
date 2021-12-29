<p align="center">
	<img width="600px" src="img/logo.png">
	<h1 align="center">BAMBOO</h1>
	<h3 align="center">Cambridge University Spaceflight</h3>
</p>

Bamboo is a Python tool that provides functions and classes for modelling the cooling systems of liquid rocket engines. Alongside this, it contains a range of other miscellaneous tools to aid with general engine design.

An introduction to the package can be found in the ['Introduction to Bamboo.ipynb'](https://github.com/cuspaceflight/bamboo/blob/master/Introduction%20to%20Bamboo.ipynb) Jupyter Notebook.

## Installation
Bamboo can be installed via pip, with the following command:

`pip install git+https://github.com/cuspaceflight/bamboo.git`

## Documentation
Made using Sphinx, available at: 
https://cuspaceflight.github.io/bamboo/

(sometimes outdated)

## Validation

One validation case has been performed so far, on the Ariane 5's Vulcain engine. It is hoped that more validation will be performed in the future.

A key effect that needs to be investigated is nucleate boiling, and how significantly that affects the results. Test cases that use a supercritical coolant will not be susceptible to nucleate boiling, and so are better modelled by Bamboo (which currently ignores two-phase effects).

|         Engine          |  Supercritical Coolant? | Peak Heat Flux Error  | Coolant Exit Temperature Error | 
|:-----------------------:|:-----------------------:|:---------------------:|:------------------------------:|
|        [Vulcain](https://github.com/cuspaceflight/bamboo/blob/master/validation/Ariane%205%20Vulcain.ipynb) |      Yes | 19.97% | 3.11% |


## Release 0.2.0
- Refactored all code to be much more user friendly and intuitive. 
- A generic heat exchanger solver has been implemented, which is now used for all simulations. This solver is more flexible than before, and allows for new features such as choice between co-flow or counter-flow cooling.
- By default, iteration is now used to find the initial inlet conditions. This removes the 'steps' in data that used to exist at the beginning of simulations.
- Any number of walls can now be added, with different materials.
- The Rao bell nozzle geometry code has been separated from the main simulation code, but is still simple to use. By default the user now inputs custom geometry.
- The mass flow rate through the engine is  automatically calculated from the geometry now (based on the throat area). It is no longer required as an input.
- The extra heat transfer due to 'fins' in the cooling channels is now modelled.

## Useful Packages
These packages are not installed with Bamboo by default, but can be very useful for creating accurate simulations.

### CoolProp
[CoolProp](https://github.com/CoolProp/CoolProp) can be used to get the thermophysical properties of huge range of fluids. It is useful for setting up the transport properties of coolants in Bamboo.

### Cantera
[Cantera](https://cantera.org/) can be used to perform equilibrium calculations and to get the thermophysical properties of ideal gases. It is useful for setting up the transport properties of the exhaust gases in Bamboo. It can also be used to calculate combustion chamber temperatures, although with the default data sets it can only do this for a limited range of fuel/oxidiser combinations.

### pypropep
[pypropep](https://github.com/jonnydyer/pypropep) can be used to calculate combustion chamber temperature and exhaust gas properties (however it cannot calculate calculate some transport properties, such as viscosity and thermal conductivity). Note that when using pypropep, the best results have been observed when you use the gas properties at the throat as your inputs into Bamboo's perfect gas model.

