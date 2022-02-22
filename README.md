<p align="center">
	<img width="600px" src="img/logo.png">
	<h1 align="center">BAMBOO</h1>
	<h3 align="center">Cambridge University Spaceflight</h3>
</p>

Bamboo is a Python tool that provides functions and classes for modelling the cooling systems of liquid rocket engines. Alongside this, it contains a range of other miscellaneous tools to aid with general engine design.

An introduction to the package can be found in the ['Introduction to Bamboo.ipynb'](https://github.com/cuspaceflight/bamboo/blob/master/Introduction%20to%20Bamboo.ipynb) Jupyter Notebook. Additional examples can be found in the ['examples'](https://github.com/cuspaceflight/bamboo/tree/master/examples) folder.

## Installation
Bamboo can be installed via pip, with the following command:

`pip install git+https://github.com/cuspaceflight/bamboo.git`

## Documentation
Made using Sphinx, available at: 
https://cuspaceflight.github.io/bamboo/

(sometimes outdated)

## Validation

All validation cases are available as Jupyter notebook, with hyperlinks below. Positive signs on the errors represent an overprediction (i.e. excess heat transfer rate, excess pressure drop, or excess temperature rise).

All validation cases were performed with the Gnielinski equation for coolant-side convection, the Bartz (sigma) equation for exhaust-side convection, and smooth walls were assumed for the pressure drop.

It can be seen that overall, Bamboo tends to overpredict temperatures and pressure drops, and so would <i>usually</i> result in a conservative design if used to design an engine.

A key effect that needs to be investigated is nucleate boiling, and how significantly that affects the results. Test cases that use a supercritical coolant will not be susceptible to nucleate boiling, and so are better modelled by Bamboo (which currently ignores two-phase effects).

|         Engine          |  Coolant State | Peak Heat Flux Error  | Coolant Temperature Rise Error |  Coolant Pressure Drop Error | 
|:-----------------------:|:-----------------------:|:---------------------:|:------------------------:|:------------------------:|
|[Vulcain Chamber](https://github.com/cuspaceflight/bamboo/blob/master/validation/Vulcain%20Combustion%20Chamber.ipynb) |Supercritical|+40.3%|+17.9%|+56.1%|
|[Vulcain Nozzle Extension](https://github.com/cuspaceflight/bamboo/blob/master/validation/Vulcain%20Nozzle%20Extension.ipynb) |Supercritical| - | +2.75% | +33.4% |
|[Pavli 1966](https://github.com/cuspaceflight/bamboo/blob/master/validation/Pavli%201966.ipynb)|Gaseous|+23.2%| +33.1% | -|

## Release 0.2.1
- Added additional validation cases.
- Coolant flow solver can now accommodate compressible coolants automatically.
- Removed fin heat transfer for now, as it was producing spurious results.
- Corrected mistakes with spiralling channel geometry.
- Swapped to using the recovery temperature instead of static temperature for the exhaust gas temperature in thermal circuits.

## Useful Packages
These packages are not installed with Bamboo by default, but can be very useful for creating accurate simulations.

### CoolProp
[CoolProp](https://github.com/CoolProp/CoolProp) can be used to get the thermophysical properties of huge range of fluids. It is useful for setting up the transport properties of coolants in Bamboo.

### Cantera
[Cantera](https://cantera.org/) can be used to perform equilibrium calculations and to get the thermophysical properties of ideal gases. It is useful for setting up the transport properties of the exhaust gases in Bamboo. It can also be used to calculate combustion chamber temperatures, although with the default data sets it can only do this for a limited range of fuel/oxidiser combinations.

### pypropep
[pypropep](https://github.com/jonnydyer/pypropep) can be used to calculate combustion chamber temperature and exhaust gas properties (however it cannot calculate calculate some transport properties, such as viscosity and thermal conductivity). Note that when using pypropep, the best results have been observed when you use the gas properties at the throat as your inputs into Bamboo's perfect gas model.

