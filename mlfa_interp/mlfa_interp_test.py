import torch
from datetime import datetime
from function_approximation.mlfa.mlfa_test_cvg import write
from function_approximation.mlfa_interp.mlfa_interp import mlfa_interp
from function_approximation.mlfa_interp.mlfa_interp_generator import Mlfa_interp_generator
import matlab.engine

def mlfa_interp_test(logfile,
                     x_test: torch.DoubleTensor,
                     y_test: torch.DoubleTensor,
                     fit_multilevel: callable,
                     Eps: list,
                     Lmin: int,
                     Lmax: int,
                     mlfa_interp_generator: Mlfa_interp_generator,
                     interp: callable,
                     alpha_0: float = 0,
                     delta_0: float = 0,
                     delta_log_0: float = 0,
                     eng: matlab.engine = None):
    """
    Test routine to performs MLFA calculations for a set of different values of eps.
    The purpose of this function is to test whether the achieved accuracy is actually bounded by eps.
    Prints results to stdout and logfile.

    Inputs:
    x_test: x values for testing
    y_test: y values for testing
    Eps: list of desired accuracies
    fit_multilevel: low-level routine that trains a multilevel function approximator of f: X -> R based on the
                    training set resulting from mlfa. Its interface is:
                        max_err = fit_multilevel(xl, yl, x_test, y_test)
    other inputs are *args_mlfa: additional user variables to be passed to mlfa
    """

    now = datetime.now().strftime("%d-%B-%Y %H:%M:%S")
    write(logfile, "\n")
    write(logfile, "**********************************************************\n")
    write(logfile, "*** MLFA_n file version 0.2     produced by            ***\n")
    write(logfile, "*** Python mlfa_test_n on %s       ***\n" %now)
    write(logfile, "**********************************************************\n")
    write(logfile, "\n")
    write(logfile, "*******************************************\n")
    write(logfile, "*** MLFA errors ***\n")
    write(logfile, "*******************************************\n")
    write(logfile, "\n")
    write(logfile, " {:<12} {:<12} {:<12}\n".format("eps", "algo acc", "test acc"))

    for eps in Eps:
        if eng is None:
            _, _, acc, xl, errl = mlfa_interp(eps, Lmin, Lmax, mlfa_interp_generator, interp, alpha_0, delta_0, delta_log_0, verbose=True)[:5]
            interp_err = fit_multilevel(xl, errl, x_test, y_test)
        else:
            _, _, acc = mlfa_interp(eps, Lmin, Lmax, mlfa_interp_generator, interp, alpha_0, delta_0, delta_log_0, eng, verbose=False)
            interp_err = fit_multilevel(x_test, y_test)
        mlfa_interp_generator.reset(eng=eng)
        write(logfile, " {:<12.3e} {:<12.3e} {:<12.3e}\n".format(eps, acc, interp_err))
