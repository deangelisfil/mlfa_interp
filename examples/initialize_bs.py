from data_generation.numerical_method.finite_difference.payoff import Call, Put
from data_generation.numerical_method.finite_difference.grid import Rustic_grid_2dim
from data_generation.numerical_method.finite_difference.bs_theta import Bs_theta
from data_generation.data_generator.bs_european_generator import Bs_european_generator
from function_approximation.mlfa.mlfa_generator import Mlfa_generator
from function_approximation.function_approximator.random_feature_approximator import Random_feature_approximator
import torch
from data_generation.model_parameter import Bs_parameter
from examples.parameters import *
from data_generation.data_generator.bs_american_put_generator import Bs_american_put_generator
from function_approximation.mlfa_interp.mlfa_interp_generator import Mlfa_sparse_interp_generator, Mlfa_tensor_interp_generator
from data_generation.numerical_method.finite_difference.bs_american_put import Bs_american_put


def initialize_bs_generator(dim: int, option_type: str, execution: str, how: str) :
    """
    Initializes a data generator of BS Call type .
    Input:
            dim:            dimension of the bs call function,
                            dim=1: (sigma) -> C(sigma)
                            dim=2: (sigma, r) -> C(sigma, r),
                            dim=4: (sigma, r, T, K) -> C(sigma, r, T, K)
            option_type:    either "put" or "call"
            execution:      either "european" or "american"
            how:            either "random" or "grid"
    """
    assert option_type in {"call", "put"}
    if option_type == "call" and execution == "european" :
        Option = Call
        Bs_generator = Bs_european_generator
    elif option_type == "put" and execution == "american" :
        Option = Put
        Bs_generator = Bs_american_put_generator
    else :
        raise ValueError("(option_type, execution) are neither (call, european) nor (put, american).")

    if dim == 1 :
        bs_parameter_min, bs_parameter_max = Bs_parameter(S_0, sigma_min, r), Bs_parameter(S_0, sigma_max, r)
        option_min, option_max = Option(T, K), Option(T, K)
    elif dim == 2 :
        bs_parameter_min, bs_parameter_max = Bs_parameter(S_0, sigma_min, r_min), Bs_parameter(S_0, sigma_max, r_max)
        option_min, option_max = Option(T, K), Option(T, K)
    elif dim == 3 :
        bs_parameter_min, bs_parameter_max = Bs_parameter(S_0, sigma_min, r_min), Bs_parameter(S_0, sigma_max, r_max)
        option_min, option_max = Option(T, K_min), Option(T, K_max)
    elif dim == 4 :
        bs_parameter_min, bs_parameter_max = Bs_parameter(S_0, sigma_min, r_min), Bs_parameter(S_0, sigma_max, r_max)
        option_min, option_max = Option(T_min, K_min), Option(T_max, K_max)
    elif dim == 5 :
        bs_parameter_min, bs_parameter_max = Bs_parameter(S_0_min, sigma_min, r_min), Bs_parameter(S_0_max, sigma_max, r_max)
        option_min, option_max = Option(T_min, K_min), Option(T_max, K_max)
    else :
        raise ValueError("dimension not supported.")
    bs_parameter_tuple = bs_parameter_min, bs_parameter_max
    option_tuple = option_min, option_max
    bs_generator = Bs_generator(bs_parameter_tuple, option_tuple, how)
    return bs_generator


def initialize_bs_call_interp(M_L, dim, how) :
    """
    Auxiliary function to initialize Mlfa_interp_generator.
    This function fixes some parameters to initialize the mlfa routines for:
    Numerical method:           Crank Nicolson scheme for the Black Scholes model
    Payoff:                     European Call
    Discretization:             Hat-weighted

    Input:
        M_L:    list with the number of time steps in the CN scheme
        dim:    see initialize_bs_generator
        how:    the discretization type the interpolation grid. Either "tensor" or "sparse"
    """
    #
    # Initialize numerical method list
    #

    # Crank-Nicolson with Rannacher-startup and hat-weighted discreization of the initial condition
    rustic_grid = Rustic_grid_2dim(0, S_max_ratio, "linear")
    bs_theta = Bs_theta(theta=0.5, rustic_grid=rustic_grid, execution="european", how="hat", rn=True)
    bs_theta_L = bs_theta.create_list_in_time(M_L)

    #
    # Initialize data generator
    #

    bs_call_generator = initialize_bs_generator(dim=dim, option_type="call", execution="european", how="grid")
    if how == "tensor" :
        constructor = Mlfa_tensor_interp_generator
    elif how == "sparse" :
        constructor = Mlfa_sparse_interp_generator
    else:
        raise ValueError("how must be either 'tensor' or 'sparse'.")
    mlfa_interp_generator = constructor(numerical_method_L=bs_theta_L, data_generator=bs_call_generator)

    return mlfa_interp_generator


def initialize_bs_american_put_interp(M_L, dim, how) :
    """
    Auxiliary function to initialize Mlfa_interp_generator.
    This function fixes some parameters to initialize the mlfa routines for:
    Numerical method:     Implicit scheme for the Black Scholes model with moving grid
                          (involves a Newton iteration at each timestep)
    Payoff:               American Put
    Discretization:       Pointwise (grid moves with strike, so no need for hat-weighted discretization)

    Input:
        M_L:    list with the number of time steps in the implicit scheme
        dim:    see initialize_bs_generator
        how:    the discretization type the interpolation grid. Either "tensor" or "sparse"
    """

    #
    # Initialize numerical method list
    #

    rustic_grid = Rustic_grid_2dim(M=0, S_max_ratio=S_max_ratio, grid_type="linear", J_M_ratio=2)
    bs_american_put = Bs_american_put(rustic_grid=rustic_grid)
    bs_american_put_L = bs_american_put.create_list_in_time(M_L=M_L)

    #
    # Initialize data generator
    #

    bs_american_put_generator = initialize_bs_generator(dim=dim, option_type="put", execution="american", how="grid")
    if how == "tensor" :
        constructor = Mlfa_tensor_interp_generator
    elif how == "sparse" :
        constructor = Mlfa_sparse_interp_generator
    else:
        raise ValueError("how must be either 'tensor' or 'sparse'.")
    mlfa_interp_generator = constructor(numerical_method_L=bs_american_put_L, data_generator=bs_american_put_generator)

    return mlfa_interp_generator
