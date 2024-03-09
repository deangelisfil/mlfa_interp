from function_approximation.mlfa.mlfa_approximator import Mse_estimator_parameter
import matlab.engine
from pathlib import Path


#
# Parameters mlfa
#

lambda_ = 1 / 4

#
# Parameters plots
#

fig_width, fig_height = 6.4, 4.8

#
# Parameters rustic grid
#

S_max_ratio = 4
V_max_ratio = 4

#
# Parameters data generator
#

S_0 = 1
S_0_min = 0.5
S_0_max = 1.5
# V_0 = 0.04  # corresponds to sigma = 0.2
V_0 = 0.09 # # corresponds to sigma = 0.3
sigma = 0.2
sigma_max = 0.4
sigma_min = 0.1
r = 0.05
r_max = 0.2
r_min = 0
# r_min = 0.01
kappa_min = 1
kappa_max = 3
# theta_min = 0.04
theta_min = 0.04
theta_max = 0.16
# omega_max = 0.3
omega_max = 0.3
omega_min = 0.1
rho_max = -0.3
rho_min = -0.7
T = 1
T_max = 1.5
T_min = 0.5
K = 0.99  # slightly misaligned with S0
K_min = 0.5
K_max = 1.5