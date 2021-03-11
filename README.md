<p align="center">
	<img width="600px" src="img/logo.png">
	<h1 align="center">BAMBOO</h1>
	<h3 align="center">Cambridge University Spaceflight</h3>
</p>

Bamboo is a Python tool that provides functions and classes for modelling the cooling systems of liquid rocket engines. Alongside this, it contains a range of other miscellaneous tools to aid with general engine design.

## Installation
`pip install git+https://github.com/cuspaceflight/bamboo.git`

## General engine tools available
- Nozzle shape calculator for Rao bell nozzles.
- Get thrust and specific impulse.
- Get gas properties (temperature and pressure) as a function of position in the nozzle.
- Estimate apogee using a simple 1D trajectory simulator.
- Optimise nozzle area ratio based on the simple trajectory simulator.

## Tools for cooling system modelling
- Add a regenerative cooling jacket to the engine.
- Add a refractory (e.g. a graphite insert) into the engine.
- Steady state heating simulations for cooling jackets on their own, or cooling jackets with refractories.

## Incomplete
- Ablative cooling system modelling.
- Time dependent cooling analysis.

## Documentation
(may be outdated)

Made using Sphinx, available at: 
https://cuspaceflight.github.io/bamboo/

