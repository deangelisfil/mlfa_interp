import sys
from pathlib import Path
path = Path.cwd().parent.parent.parent.absolute()
sys.path.append(str(path))
from examples.bs.initialize_bs import initialize_bs_american_put_interp
import numpy as np
from function_approximation.mlfa_interp.mlfa_interp_test_cvg import mlfa_interp_test_cvg
from function_approximation.mlfa_interp.interp import sparse_grid_interp, generate_fit_multilevel, tensor_grid_interp
from function_approximation.mlfa_interp.mlfa_interp_plot_cvg import mlfa_interp_read_cvg, mlfa_interp_plot_cvg
from function_approximation.mlfa_interp.mlfa_interp_test_complexity import mlfa_interp_test_complexity
from function_approximation.mlfa_interp.mlfa_interp_test import mlfa_interp_test
from function_approximation.mlfa_interp.mlfa_interp_plot_complexity import mlfa_interp_plot_complexity
from function_approximation.mlfa_interp.mlfa_interp_plot import mlfa_interp_plot
from auxiliary_function.initialize_matlab_engine import initialize_matlab_engine
from function_approximation.mlfa_interp.mlfa_interp import mlfa_interp
from fractions import Fraction

if __name__ == "__main__" :

    #
    # Initialize Matlab engine
    #

    eng = initialize_matlab_engine(2)
    eng.warning('off', 'MATLAB:spinterp:insufficientDepth')
    eng.eval("options = spset("
             "'Vectorized','on', "
             "'AbsTol', 1e-16, 'RelTol', 1e-16, "
             # "'GridType', 'Clenshaw-Curtis', "
             "'GridType','Maximum',"
             "'KeepFunctionValues','on','KeepGrid','on');", nargout=0)

    #
    # initialize the mlfa_interp_generator and the function handle
    #

    dim = 2
    M_L = np.array([8, 16, 32, 64], dtype=int)
    how = "tensor"
    mlfa_interp_generator = initialize_bs_american_put_interp(M_L=M_L, dim=dim, how=how)

    eng.workspace['M_L'] = M_L
    eng.workspace['dim'] = dim
    eng.workspace['how'] = how
    eng.eval("mlfa_interp_generator = py.examples.bs.initialize_bs.initialize_bs_american_put_interp(M_L, dim, how);",nargout=0)
    eng.workspace['target'] = "delta_f"
    eng.eval("f = wrapper(mlfa_interp_generator, target);", nargout=0)
    eng.eval("low_high_tuple = mlfa_interp_generator.data_generator.get_low_high_tuple(py.True);"
             "range = double(py.numpy.array(low_high_tuple).T);", nargout=0)
    eng.eval("z = cell(1, length(M_L));", nargout=0)

    #
    # Initialize mlfa function
    #

    alpha = 2
    delta = 2
    # delta = 2 / dim
    # delta_log = 3 * (dim - 1)
    delta_log = 0
    factor_N = 1
    Lmin = 1
    Lmax = 7
    mlfa_args = [Lmin,
                 Lmax,
                 mlfa_interp_generator,
                 sparse_grid_interp,
                 alpha,
                 delta,
                 delta_log,
                 factor_N,
                 eng]
    filename_cvg = "mlfa_interp_test_cvg_bs_american_put_{0}_dim{1}.txt".format(how, dim)
    filename_complexity = "mlfa_interp_test_complexity_bs_american_put_{0}_dim{1}.txt".format(how, dim)
    filename_acc = "mlfa_interp_test_n_bs_american_put_{0}_dim{1}.txt".format(how, dim)

    #
    # Convergence test
    #

    L = 3
    K = 10

    logfile = open(filename_cvg, "w")
    mlfa_interp_test_cvg(logfile=logfile,
                         L=L,
                         K=K,
                         mlfa_interp_generator = mlfa_interp_generator,
                         interp = sparse_grid_interp,
                         L0=0,
                         eng=eng)
    logfile.close()

    cost_exp = Fraction(1)
    mlfa_interp_plot_cvg(filename_cvg, gamma=2, nvert=2, save=True)

    #
    # Complexity
    #

    Eps = [5e-3, 2e-3, 1e-3]
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
    Eps = [2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5]
    fit_multilevel = generate_fit_multilevel(eng = eng)
    logfile = open(filename_acc, "w")
    mlfa_interp_test(logfile, x_test, y_test, fit_multilevel, Eps, *mlfa_args)
    logfile.close()

    mlfa_interp_plot(filename_acc, save=True)

    eng.quit()