import sys
from pathlib import Path
path = Path.cwd().parent.parent.parent.absolute()
sys.path.append(str(path))
from function_approximation.mlfa_interp.mlfa_interp import mlfa_interp
from examples.bs.initialize_bs import initialize_bs_call_interp
from function_approximation.mlfa_interp.interp import tensor_grid_interp
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from auxiliary_function.error import maxe
from function_approximation.mlfa_interp.mlfa_interp_test_cvg import mlfa_interp_test_cvg
from function_approximation.mlfa_interp.interp import tensor_grid_interp, generate_fit_multilevel
from function_approximation.mlfa_interp.mlfa_interp_plot_cvg import mlfa_interp_read_cvg, mlfa_interp_plot_cvg
from function_approximation.mlfa_interp.mlfa_interp_test_complexity import mlfa_interp_test_complexity
from function_approximation.mlfa_interp.mlfa_interp_test import mlfa_interp_test
from function_approximation.mlfa_interp.mlfa_interp_plot_complexity import mlfa_interp_plot_complexity
from function_approximation.mlfa_interp.mlfa_interp_plot import mlfa_interp_plot
from fractions import Fraction


if __name__ == "__main__" :

    #
    # Initialize mlfa function
    #

    dim = 3
    M_L = [2, 4, 8, 16, 32, 64, 128, 256]
    mlfa_interp_generator = initialize_bs_call_interp(M_L=M_L, dim=dim, how="tensor")

    alpha = 2
    delta = 2 / dim
    Lmin = 1
    Lmax = 7
    mlfa_args = [Lmin,
                 Lmax,
                 mlfa_interp_generator,
                 tensor_grid_interp,
                 alpha,
                 delta]
    filename_cvg = "mlfa_interp_test_cvg_bs_call_tensor_dim{}.txt".format(dim)
    filename_complexity = "mlfa_interp_test_complexity_bs_call_tensor_dim{}.txt".format(dim)
    filename_acc = "mlfa_interp_test_n_bs_call_tensor_dim{}.txt".format(dim)

    #
    # Convergence test
    #

    L = 7
    K = 6

    logfile = open(filename_cvg, "w")
    mlfa_interp_test_cvg(logfile=logfile,
                         L=L,
                         K=K,
                         mlfa_interp_generator = mlfa_interp_generator,
                         interp = tensor_grid_interp,
                         L0=0)
    logfile.close()

    cost_exp = Fraction(3/2)
    mlfa_interp_plot_cvg(filename_cvg, gamma=2, nvert=3, save=True, cost_exp=cost_exp, filename_complexity=filename_complexity)


    #
    # Complexity
    #

    Eps = [2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5]
    Lstarts = 1 * np.ones(len(Eps), dtype=int)  # starting level is 1 across accuracies
    logfile = open(filename_complexity, "w")
    mlfa_interp_test_complexity(filename_cvg, logfile, Eps, *mlfa_args)
    logfile.close()

    mlfa_interp_plot_complexity(filename_complexity, save=True, cost_exp=1.5)

    #
    # Accuracy test
    #

    np.random.seed(0)
    N_test = int(1e6)
    x_test, y_test = mlfa_interp_generator.data_generator.test(N=N_test)
    Eps = [2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5]
    fit_multilevel = generate_fit_multilevel(get_grid=mlfa_interp_generator.data_generator.get_grid,
                                             grid_interp=tensor_grid_interp)
    logfile = open(filename_acc, "w")
    mlfa_interp_test(logfile, x_test, y_test, fit_multilevel, Eps, *mlfa_args)
    logfile.close()

    mlfa_interp_plot(filename_acc, save=True)