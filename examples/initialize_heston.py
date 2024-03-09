from data_generation.numerical_method.finite_difference.payoff import *
from data_generation.numerical_method.finite_difference.grid import Rustic_grid_3dim
from data_generation.numerical_method.finite_difference.heston_cn import Heston_cn
from data_generation.model_parameter import Heston_parameter
from data_generation.data_generator.heston_call_generator import Heston_call_generator
from function_approximation.mlfa.mlfa_generator import Mlfa_generator
from function_approximation.function_approximator.random_feature_approximator import Random_feature_approximator
import torch
from examples.parameters import *
from function_approximation.mlfa_interp.mlfa_interp_generator import Mlfa_sparse_interp_generator


def initialize_heston_call_generator(how: str):
    """
    Initializes a Data generator of Heston call type
    """
    heston_parameter_min = Heston_parameter(S_0=S_0,
                                            V_0=V_0,
                                            r=r_min,
                                            kappa=kappa_min,
                                            theta=theta_min,
                                            omega=omega_min,
                                            rho=rho_min)
    heston_parameter_max = Heston_parameter(S_0=S_0,
                                            V_0=V_0,
                                            r=r_max,
                                            kappa=kappa_max,
                                            theta=theta_max,
                                            omega=omega_max,
                                            rho=rho_max)
    heston_parameter_tuple = heston_parameter_min, heston_parameter_max
    call_min, call_max = Call(T, K_min), Call(T, K_max)
    call_tuple = call_min, call_max
    heston_call_generator = Heston_call_generator(heston_parameter_tuple, call_tuple, how=how)

    return heston_call_generator


def initialize_heston_call_sparse_interp(M_L: np.array):
    """
    Initializes a sparse grid interpolator for Heston PDE with call payoff.
    """

    #
    # Initialize numerical method list
    #

    # hat-weighted discretization of the initial condition
    # Heston_cn uses a Crank-Nicolson scheme with four half-time steps of RN startup.
    rustic_grid = Rustic_grid_3dim(M=0,
                                   S_max_ratio=S_max_ratio,
                                   V_max_ratio=V_max_ratio,
                                   grid_type="linear",
                                   J_M_ratio=8,
                                   K_M_ratio=1)  # M=0 not used
    heston_cn = Heston_cn(rustic_grid, how="hat")
    heston_cn_L = heston_cn.create_list_in_time(M_L)

    #
    # Initialize data generator
    #

    heston_call_generator = initialize_heston_call_generator(how="grid")
    mlfa_interp_generator = Mlfa_sparse_interp_generator(numerical_method_L=heston_cn_L,
                                                         data_generator=heston_call_generator)

    return mlfa_interp_generator



