"""
Temporary file for testing nucleate boiling equations

References:
 - [1] - Rohsenow, Handbook of Heat Transfer
"""
GRAVITY = 9.81

n = 1.7     # n = 1 for water, 1.7 for other fluids
RHS = C_sf * ( 1 / (mu_l * i_lg) * (sigma / (GRAVITY * (rho_l - rho_g)))**0.5 )**0.33 * (cp_l * mu_l / k_l)**n  # Equation (15.81)
q_R = cp_l * dT_sat / (i_lg * RHS)

q_FC = # Forced convection

q_D?

q = (q_FC**2 + (q_R * (1 - q_D/q_R) )**2 )**0.5             # Equation (15.237)