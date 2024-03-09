import numpy as np
from function_approximation.mlfa_interp.mlfa_interp import mlfa_interp
from function_approximation.mlfa.mlfa_generator import Mlfa_generator
from data_generation.data_generator.data_generator_abstract import Data_generator_abstract
from function_approximation.mlfa.mlfa_test_cvg import write
from function_approximation.mlfa_interp.mlfa_interp_plot_cvg import mlfa_interp_read_cvg
from function_approximation.mlfa_interp.mlfa_interp_generator import Mlfa_interp_generator, Mlfa_tensor_interp_generator, Mlfa_sparse_interp_generator
from examples.parameters import lambda_
import matlab.engine
from numpy import abs, log
from scipy.optimize import fsolve, bisect

def mlfa_interp_test_complexity(filename,
                                logfile,
                                Eps: list,
                                Lmin: int,
                                Lmax: int,
                                mlfa_interp_generator: Mlfa_interp_generator,
                                interp: callable,
                                alpha_0: int = 0,
                                delta_0: int = 0,
                                delta_log_0: int = 0,
                                factor_N: int = 1,
                                eng: matlab.engine = None):
    """
    Multilevel Interpolation test routine. Prints results to stdout and file.
    todo: write description
    """

    write(logfile, "\n")
    write(logfile, "***************************** \n")
    write(logfile, "*** MLFA complexity tests *** \n")
    write(logfile, "***************************** \n\n")
    write(logfile, "   eps      acc         mlfa_cost   std_cost   savings     N_l \n")
    write(logfile, "-------------------------------------------------------------- \n")

    how, dim, _, _, _, B, interp_yl, _, cost, L0, Nn_train, interp_errn, alpha, _, _, delta = \
        mlfa_interp_read_cvg(filename)

    # use the input alpha_0 and delta_0 if provided
    alpha = alpha_0 if alpha_0 > 0 else alpha
    delta = delta_0 if delta_0 > 0 else delta
    delta_log = delta_log_0 if delta_log_0 > 0 else 0

    for eps in Eps:
        if mlfa_interp_generator.Lstart != 0:
            raise NotImplementedError("The version with Lstart != 0 has not been implemented yet.")
        mlfa_interp_args = [Lmin, Lmax, mlfa_interp_generator, interp, alpha, delta, delta_log, factor_N, eng]
        Nl, Cl, acc = mlfa_interp(eps, *mlfa_interp_args)
        mlfa_cost = np.dot(Nl, Cl)
        # uses acc NOT eps for std_cost!
        assert L0 == 0, "L0 must be 0 for the cost estimation."
        # uses the fact that the interpolation error of f_0 is roughly the same as the one of f_L
        # set m to be the biggest index in Nn_train for which interp_errn >= (1 - lambda_) * acc
        n = np.where(interp_errn >= (1 - lambda_) * acc)[0][-1]
        if how == "tensor":
            N = Nn_train[n] * (interp_errn[n] / ((1 - lambda_) * acc)) ** (1 / delta)
        elif how == "sparse":
            N_fun = lambda N : \
                interp_errn[n] \
                * (Nn_train[n] / N) ** delta \
                * (abs(log(N) + log(factor_N)) / abs(log(Nn_train[n]) + log(factor_N))) ** delta_log \
                - (1 - lambda_) * acc
            N = fsolve(N_fun, x0=Nn_train[n], xtol=1e-4)[0]
        else:
            raise NotImplementedError("Either Mlfa_tensor_interp_generator or Mlfa_sparse_interp_generator.")
        std_cost = Cl[-1] * N
        mlfa_interp_generator.reset(eng=eng)

        write(logfile, "%.3e   %.3e   %.3e   %.3e   %7.2f " % (eps, acc, mlfa_cost, std_cost, std_cost / mlfa_cost))
        write(logfile, " ".join(["%9d " % n for n in Nl]))
        write(logfile, "\n")
    write(logfile, "\n")