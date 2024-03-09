from function_approximation.mlfa_interp.mlfa_interp_generator import Mlfa_interp_generator, Mlfa_tensor_interp_generator, Mlfa_sparse_interp_generator
import numpy as np
from datetime import datetime
from auxiliary_function.error import maxe
from function_approximation.mlfa.mlfa_test_cvg import write
import matlab.engine


def mlfa_interp_test_cvg(logfile,
                         L: int,
                         K: int,
                         mlfa_interp_generator: Mlfa_interp_generator,
                         interp: callable,
                         L0: int = 0,
                         eng: matlab.engine = None):
    """

    Multilevel Interpolation test routine for determining the convergence rates of the meta-theorem.
    Prints results to stdout and file.

    If the user is interested in the estimation of alpha, beta and gamma only, use K = mlfa_interp_generator.kl[0]
    todo: implement this. At the moment, it will throw an error.

    Inputs:
        L:  number of levels of the finite difference approximation for convergence tests
        K:  number of levels of the interpolation grid for convergence tests
        mlfa_interp_generator: The user low-level routine for the generation of level l data on the interpolation grid.
            Calling the function automatically increases the accuracy level of the interpolation grid by 1. Its interface is

            grid, y, err, x_val, y_val, err_val, cost = mlfa_generator(l)

            Inputs:
                l: level

            Outputs:
                grid: interpolation grid X_{k-1} of level k-1 on the parameter space
                y: numerical approximations of f_l(x) for x in X_{k-1}
                err: numerical approximation for x in X_{k-1} of
                f_0(x)                  on level l = 0
                f_l(x) - f_{l-1}(x)     on level l > 0
                x_val: interpolation grid X_k of level k on the parameter space.
                    In particular, X_{k-1} is a subset of X_k.
                y_val: numerical approximations of f_l(x) for x in X_k
                err_val: numerical approximation for x in X_k of
                f_0(x)                  on level l = 0
                f_l(x) - f_{l-1}(x)     on level l > 0
                cost: cost to generate err_val

        interp: the user low-level routine for the estimation of the interpolation error with grid or sparse grids.
            Its interface is

            max_error = interp(grid, y, x_val, y_val)

            Inputs:
                grid: interpolation grid
                y: interpolation values
                x_val: validation points
                y_val: validation values
            Ouput:
                interp_err: maximum error of the intepolation, where the error is estimated at the validation points.

        logfile: file handle for printing to file

        # L0: fixed level for the convergence test of mse w.r.t. sample size.

    Note:
        The number of fixed samples N0 for the convergence tests w.r.t. level is specified in mlfa_interp_generator.kl[0].
    """

    now = datetime.now().strftime("%d-%B-%Y %H:%M:%S")
    write(logfile, "\n")
    write(logfile, "**********************************************************\n")
    write(logfile, "*** MLFA file version 0.2     produced by              ***\n")
    write(logfile, "*** Python mlfa_interp_test_cvg on %s       ***\n" % now)
    write(logfile, "**********************************************************\n")
    write(logfile, "\n")
    write(logfile, "*************\n")
    write(logfile, "*** d = %d ***\n" % mlfa_interp_generator.data_generator.get_dim(drop_constant_parameter=True))
    write(logfile, "*************\n")
    write(logfile, "\n")

    if isinstance(mlfa_interp_generator, Mlfa_tensor_interp_generator):
        operator = "I"
    elif isinstance(mlfa_interp_generator, Mlfa_sparse_interp_generator):
        operator = "A"
    else:
        raise NotImplementedError("For now, only tensor and sparse grid interpolation is implemented.")

    #
    # First, test the cvg. of the f.d. error, of the interpolation of the f.d. error and of the computational cost
    # calculate ||\Delta f_l||_\infty for l=0,...,L
    # calculate ||(I-I_{k0})[\Delta f_l]||_\infty for l=0,...,L, with k0 = mlfa_interp_generator.kl[l]
    # all the errors are validated on the interpolation grid k0 but the training is done on k0-1.
    #

    assert np.array_equal(mlfa_interp_generator.get_kl(),
                          np.ones(len(mlfa_interp_generator.get_kl()), dtype=int) * mlfa_interp_generator.kstart), \
           "the initial discretization of the interpolation grids is not the same for all levels."
    Lstart = mlfa_interp_generator.Lstart

    max_yl = np.zeros(L + 1)
    max_errl = np.zeros(L + 1)
    cost = np.zeros(L + 1)
    interp_yl = np.zeros(L + 1)
    interp_errl = np.zeros(L + 1)
    Nl = np.zeros(L + 1)
    Nl_train = np.zeros(L + 1)
    for l in range(Lstart, L + 1) :
        # todo: add kurtosis check and consistency check
        if isinstance(mlfa_interp_generator, Mlfa_tensor_interp_generator):
            grid_train, y_train, err_train, x, y, err, cst = mlfa_interp_generator(l=l)
            interp_yl[l] = interp(grid=grid_train, y=y_train, x_val=x, y_val=y)
            interp_errl[l] = interp(grid=grid_train, y=err_train, x_val=x, y_val=err)
            Nl_train[l] = len(y_train)
            Nl[l] = len(y)
            max_yl[l] = maxe(y)
            max_errl[l] = maxe(err)
        else:
            # Mlfa_sparse_interp_generator
            Nl_train[l], Nl[l], max_errl[l], cst = mlfa_interp_generator(l=l, eng=eng)
            interp_errl[l] = interp(eng=eng)
        cost[l] = cst / Nl[l]

    # generate the values for y in the sparse grid case
    if isinstance(mlfa_interp_generator, Mlfa_sparse_interp_generator):
        mlfa_interp_generator.reset(eng=eng) # reset the generator
        eng.workspace['target'] = "f" # change target function
        eng.eval("f = wrapper(mlfa_interp_generator, target);", nargout=0)
        for l in range(Lstart, L + 1):
            _, _, max_yl[l], _ = mlfa_interp_generator(l=l, eng=eng)
            interp_yl[l] = interp(eng=eng)
        eng.workspace['target'] = "delta_f" # change target function back to delta_f
        eng.eval("delta_f = wrapper(mlfa_interp_generator, target);", nargout=0)

    N0 = Nl[-1]
    N0_train = Nl_train[-1]

    #
    # Second, use linear regression to estimate alpha, beta, gamma
    #

    assert L > 2, "Need L > 2 to estimate alpha, beta, gamma"
    L1 = 2
    L2 = L + 1
    pa = np.polyfit(range(L1, L2), np.log2(max_errl[L1:L2]), 1); alpha = -pa[0]
    pb = np.polyfit(range(L1, L2), np.log2(interp_errl[L1:L2]), 1); beta = -pb[0]
    pg = np.polyfit(range(L1, L2), np.log2(cost[L1:L2]), 1); gamma = pg[0]

    #
    # Third, test the convergence of the interpolation error
    # calculate ||(I-I_k)[f_L0]||_\infty for k0<=k<=K, with k0 = mlfa_interp_generator.kl[0]
    #

    mlfa_interp_generator.reset(eng=eng)
    Nn = []
    Nn_train = []
    interp_errn = []
    while mlfa_interp_generator.get_kl()[L0] <= K:
        if isinstance(mlfa_interp_generator, Mlfa_tensor_interp_generator):
            grid_train, _, err_train, x, _, err, _ = mlfa_interp_generator(l=L0)
            N_train = len(err_train)
            N = len(err)
            interp_err = interp(grid=grid_train, y=err_train, x_val=x, y_val=err)
        else:
            # Mlfa_sparse_interp_generator
            N_train, N, _, _ = mlfa_interp_generator(l=L0, eng=eng)
            interp_err = interp(eng=eng)
        Nn_train.append(N_train)
        Nn.append(N)
        interp_errn.append(interp_err)

    #
    # Fourth, use linear regression to estimate delta
    #

    assert len(interp_errn) > 1, "Need K - k_0 >= 1 to estimate delta"
    K0 = 1 # todo: optimize this
    pd = np.polyfit(np.log2(Nn_train[K0:]), np.log2(interp_errn[K0:]), 1); delta = -pd[0]

    #
    # Fifth, write the results in logfile
    #

    write(logfile, "************************************************************************\n")
    write(logfile, "*** Convergence tests w.r.t. levels                                  ***\n")
    write(logfile, "*** using N =%6d samples (B =%4d training samples)               ***\n" % (N0, N0_train))
    write(logfile, "************************************************************************\n")
    write(logfile, "\n")
    write(logfile,f" l    ||f_l||       ||\Delta f_l||   ||(I-{operator}_B)[f_l]||   ||(I-{operator}_B)[\Delta f_l]||")
    write(logfile, "   kurtosis   check   cost   \n-----------------------------")
    write(logfile, "------------------------------------------------------------------------------")
    write(logfile, "\n")
    for l in range(L+1):
        write(logfile, " %-2d   %-11.3e   %-14.3e   %-16.3e   %-23.3e   --------   -----   %-7.2e \n" % \
              (l, max_yl[l], max_errl[l], interp_yl[l], interp_errl[l], cost[l]))

    write(logfile, "\n")
    write(logfile, "**************************************************\n")
    write(logfile, "*** Convergence tests w.r.t. number of samples ***\n")
    write(logfile, "*** using level l = %2d                         ***\n" % L0)  # L0 is set to 0
    write(logfile, "**************************************************\n")
    write(logfile, "\n")
    write(logfile, f" N        N_train   ||(I-{operator}_N)[\Delta f_l]||              \n")
    write(logfile, "--------------------------------------------------\n")
    for n in range(len(Nn)) :
        write(logfile, " %-6d   %-7d   %-15.3e \n" % (Nn[n], Nn_train[n], interp_errn[n]))

    write(logfile, "\n")
    write(logfile, "******************************************************\n")
    write(logfile, "*** Linear regression estimates of MLFA parameters ***\n")
    write(logfile, "******************************************************\n")
    write(logfile, "\n")
    write(logfile, " alpha     = %f  (exponent of finite difference error) \n" % alpha)
    write(logfile, " beta      = %f  (exponent of interpolation error w.r.t. level) \n" % beta)
    write(logfile, " gamma     = %f  (exponent of cost) \n" % gamma)
    write(logfile, " delta     = %f  (exponent of interpolation error w.r.t. number of samples) \n" % delta)