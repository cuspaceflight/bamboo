# bamboo
Nozzle flow dynamics using perfect gas equations for 1D isentropic flow  

The intention is that the library uses as many analytical (as opposed to numerical) methods as possible, so the code can run quickly and can be used for optimisation and design studies.

## Currently includes
- Nozzle shape calculator for Rao bell nozzles.
- Get thrust and specific impulse.
- Get gas properties (temperature and pressure) as a function of position in the nozzle.
- Estimate apogee using a simple 1D trajectory simulator.

## Incomplete:
- Optimise nozzle area ratio based on a simple flight trajectory simulator.
