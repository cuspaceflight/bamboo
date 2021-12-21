"""
Temporary file for testing bamboo 0.2.0
"""

import bamboo as bam
import numpy as np
import matplotlib.pyplot as plt

xs, ys = bam.rao.get_rao_contour(Rc = 0.045, Rt = 0.02, area_ratio = (0.025/0.02)**2, Lc = 0.10, theta_conv = 45)

plt.plot(xs, ys)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Rao Bell Nozzle")
plt.axis('equal')
plt.grid()
plt.show()

xs, ys = bam.rao.get_rao_contour(Rc = 0.045, Rt = 0.02, area_ratio = (0.05/0.02)**2, Lc = 0.10, theta_conv = 45)

plt.plot(xs, ys)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Rao Bell Nozzle")
plt.axis('equal')
plt.grid()
plt.show()