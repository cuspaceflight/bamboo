"""
Rao bell nozzle data.

References:
    - [1] - The Thrust Optimised Parabolic nozzle, AspireSpace, http://www.aspirespace.org.uk/downloads/Thrust%20optimised%20parabolic%20nozzle.pdf
"""

import numpy as np

def rao_theta_n(area_ratio, length_fraction = 0.8):
    """Returns the contour angle at the inflection point of the bell nozzle, by interpolating data.   
    Data obtained by using http://www.graphreader.com/ on the graph in Reference [1].

    Args:
        area_ratio (float): Area ratio of the nozzle (A2/At)
        length_fraction (int, optional): Nozzle contraction percentage, as defined in Reference [1]. Defaults to 0.8.
    
    Returns:
        float: "theta_n", angle at the inflection point of the bell nozzle (rad)
    """
    
    #Choose the data to use
    if length_fraction == 0.8:
        data = {"area_ratio":[3.678,3.854,4.037,4.229,4.431,4.642,4.863,5.094,5.337,5.591,5.857,6.136,6.428,6.734,7.055,7.391,7.743,8.111,8.498,8.902,9.326,9.77,10.235,10.723,11.233,11.768,12.328,12.915,13.53,14.175,14.85,15.557,16.297,17.074,17.886,18.738,19.63,20.565,21.544,22.57,23.645,24.771,25.95,27.186,28.48,29.836,31.257,32.746,34.305,35.938,37.649,39.442,41.32,43.288,45.349,47.508,49.77,52.14,54.623],
                "theta_n":[21.067,21.319,21.601,21.908,22.215,22.482,22.734,22.986,23.238,23.489,23.736,23.984,24.232,24.48,24.728,24.965,25.176,25.387,25.598,25.809,26.02,26.231,26.441,26.617,26.792,26.968,27.143,27.319,27.494,27.67,27.845,27.996,28.134,28.272,28.409,28.547,28.684,28.822,28.965,29.119,29.272,29.426,29.58,29.733,29.887,30.04,30.169,30.298,30.426,30.554,30.683,30.811,30.94,31.085,31.239,31.393,31.546,31.7,31.853]}
    else:
        raise ValueError("The length percent given does not match any of the available data.")
    
    #Make sure we're not outside the bounds of our data
    if area_ratio < 3.7 or area_ratio > 47:
        raise ValueError(f"The area ratio provided ({area_ratio}) is outside of the range of available data. Maximum available is {data['area_ratio'][-1]}, minimum is {data['area_ratio'][0]}.")
    
    else:
        #Linearly interpolate and return the result, after converting it to radians.
        return np.interp(area_ratio, data["area_ratio"], data["theta_n"]) * np.pi/180

def rao_theta_e(area_ratio, length_fraction = 0.8):
    """Returns the contour angle at the exit of the bell nozzle, by interpolating data.  
    Data obtained by using http://www.graphreader.com/ on the graph in Reference [1].

    Note:
        Not actually used by Bamboo (a quadratic only has three degrees of freedom, so Bamboo fits the nozzle contour quadratic to the inflection position, inflection angle, and exit position).

    Args:
        area_ratio (float): Area ratio of the nozzle (A2/At)
        length_fraction (int, optional): Nozzle contraction percentage, as defined in Reference [1]. Defaults to 0.8.
    
    Returns:
        float: "theta_e", angle at the exit of the bell nozzle (rad)
    """
    
    #Choose the data to use
    if length_fraction == 0.8:
       data = {"area_ratio":[3.678,3.854,4.037,4.229,4.431,4.642,4.863,5.094,5.337,5.591,5.857,6.136,6.428,6.734,7.055,7.391,7.743,8.111,8.498,8.902,9.326,9.77,10.235,10.723,11.233,11.768,12.328,12.915,13.53,14.175,14.85,15.557,16.297,17.074,17.886,18.738,19.63,20.565,21.544,22.57,23.645,24.771,25.95,27.186,28.48,29.836,31.257,32.746,34.305,35.938,37.649,39.442,41.32,43.288,45.349,47.508],
               "theta_e":[14.355,14.097,13.863,13.624,13.372,13.113,12.889,12.684,12.479,12.285,12.096,11.907,11.733,11.561,11.393,11.247,11.101,10.966,10.832,10.704,10.585,10.466,10.347,10.229,10.111,10.001,9.927,9.854,9.765,9.659,9.553,9.447,9.341,9.235,9.133,9.047,8.962,8.877,8.797,8.733,8.67,8.602,8.5,8.398,8.295,8.252,8.219,8.187,8.155,8.068,7.96,7.851,7.744,7.68,7.617,7.553]}
            
    else:
        raise ValueError("The length percent given does not match any of the available data.")
    
    #Check if we're outside the bounds of our data
    if area_ratio < 3.7 or area_ratio > 47:
        raise ValueError(f"The area ratio provided ({area_ratio}) is outside of the range of available data. Maximum available is {data['area_ratio'][-1]}, minimum is {data['area_ratio'][0]}.")
    
    else:
        #Linearly interpolate and return the result, after converting it to radians
        return np.interp(area_ratio, data["area_ratio"], data["theta_e"]) * np.pi/180

def get_rao_geometry(Rc, Rt, Re, L_c):
    area_ratio = Re**2 / Rt**2

    theta_n = rao_theta_n(area_ratio = area_ratio)
    theta_e = rao_theta_e(area_ratio = area_ratio)

    # Need to copy out the equations from Reference [1]