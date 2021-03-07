<p align="center">
	<img width="600px" src="img/logo.png">
	<h1 align="center">BAMBOO</h1>
	<h3 align="center">Cambridge University Spaceflight</h3>
</p>

Liquid rocket engine modelling package, using 1D compressible flow relations and assuming perfect gases where possible.

## Installation
`pip install git+https://github.com/cuspaceflight/bamboo.git`

## Currently includes
- Nozzle shape calculator for Rao bell nozzles.
- Get thrust and specific impulse.
- Get gas properties (temperature and pressure) as a function of position in the nozzle.
- Estimate apogee using a simple 1D trajectory simulator.
- Optimise nozzle area ratio based on the simple trajectory simulator.
- Tools for modelling engine cooling systems.
- Tools for modelling stress in the engine walls.

## Incomplete
- Ablative cooling system modelling
- Time dependent cooling analysis

## Documentation
(currently outdated)

Made using Sphinx, available at: 
https://cuspaceflight.github.io/bamboo/

