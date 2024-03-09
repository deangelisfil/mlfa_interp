import numpy as np
from mlfa_interp.mlfa_interp_generator import Mlfa_interp_generator, Mlfa_tensor_interp_generator, Mlfa_sparse_interp_generator
from auxiliary_function.error import maxe
from examples.parameters import lambda_
from numpy import log
import matlab.engine


class FiniteDifferenceConvergenceFailure(Exception):
    pass


def mlfa_interp(eps: float,
                Lmin: int,
                Lmax: int,
                mlfa_interp_generator: Mlfa_interp_generator,
                interp: callable,
                alpha_0: float = 0,
                delta_0 = 0,
                delta_log_0 = 0,
                factor_N = 1,
                eng: matlab.engine = None,
                verbose: bool = False):
    """
    Multilevel function estimation of f: X -> R.

    (xl, errl_train) = mlfa(...)

    Inputs:
    eps:  desired accuracy (rms error) >  0
    Lmin: minimum level of refinement  >= 1
    Lmax: maximum level of refinement  >= Lmin
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

    alpha -> numerical error is O(2^{-alpha*l})
    delta -> sampling error is O(N^{-delta*l})

    Outputs:
    xl: interpolation grid on each level
    errl_train: f_l(x) - f_{l-1}(x) for x on the interpolation grid if each level
    """

    # Check arguments
    if alpha_0 <= 0 and Lmin < 2:
        raise ValueError("Need Lmin >= 2 to estimate alpha, beta and gamma.")
    if Lmax < Lmin:
        raise ValueError("Need Lmax >= Lmin")
    if eps <= 0:
        raise ValueError("Need eps > 0")

    # Initialization
    L = Lmin
    alpha = max(0., alpha_0)
    delta = max(0., delta_0)
    delta_log = max(0., delta_log_0)

    xl = [np.array([]) for _ in range(L + 1)] # only returned if verbose
    errl = [np.array([]) for _ in range(L + 1)] # only returned if verbose

    costl = np.zeros(L + 1)
    max_errl = np.zeros(L + 1)
    interp_errl = np.zeros(L + 1)
    Cl = np.zeros(L + 1)
    train_interp_errl = np.zeros(L + 1)

    Nl = np.zeros(L + 1, dtype=int)
    Nl_train = np.zeros(L + 1, dtype=int)

    refinel = [True for _ in range(L + 1)]

    while sum(refinel) > 0:
        print("refinel:", refinel)
        for l, refine in enumerate(refinel):
            if refine:
                if isinstance(mlfa_interp_generator, Mlfa_tensor_interp_generator):
                    # generate (extra) samples
                    grid_train, _, err_train, xl[l], _, errl[l], costl[l] = mlfa_interp_generator(l)
                    # interpolate
                    train_interp_errl[l] = interp(grid=grid_train, y=err_train, x_val=xl[l], y_val=errl[l])
                    Nl_train[l] = err_train.size
                    Nl[l] = errl[l].size
                    max_errl[l] = maxe(errl[l])
                elif isinstance(mlfa_interp_generator, Mlfa_sparse_interp_generator):
                    Nl_train[l], Nl[l], max_errl[l], costl[l] = mlfa_interp_generator(l=l, eng=eng)
                    train_interp_errl[l] = interp(eng=eng)
                interp_errl[l] = train_interp_errl[l] \
                                 * (Nl_train[l] / Nl[l]) ** delta \
                                 * (np.abs(log(Nl[l]) + log(factor_N)) / np.abs(log(Nl_train[l]) + log(factor_N))) ** delta_log
                Cl[l] = costl[l] / Nl[l]
                refinel[l] = False

        # check interpolation convergence
        print("Nl:", Nl, "Nl_train:", Nl_train, flush=True)
        print("training interpolation error", train_interp_errl, flush=True)
        print("extrapolated interpolation error", interp_errl, flush=True)

        # check finite difference error
        fd_err = max_errl[L] / (2 ** alpha - 1)
        print("estimated finite difference error:", fd_err, flush=True)
        if fd_err > lambda_ * eps:
            if L == Lmax :
                raise FiniteDifferenceConvergenceFailure("Failed to achieve finite difference convergence")
            else :
                L = L + 1
                refinel.append(True)
                costl, xl, errl, Nl, Nl_train, max_errl, interp_errl, Cl, train_interp_errl = \
                    append_to_arrays(costl, xl, errl, Nl, Nl_train, max_errl, interp_errl, Cl, train_interp_errl)
        # check interpolation error
        elif interp_errl.sum() > (1 - lambda_) * eps:
            lstar = np.argmax(interp_errl / (Nl * Cl))
            refinel[lstar] = True

    acc = fd_err + interp_errl.sum()
    print("Achieved accuracy:", acc, flush=True)

    if verbose:
        assert eng is None, "eng must be None if verbose is True."
        return Nl, Cl, acc, xl, errl, interp_errl
    else:
        return Nl, Cl, acc

def append_to_arrays(*arrays):
    """
    Append default values to multiple arrays.
    """
    res = []
    for array in arrays:
        if isinstance(array, list):
            array.append(np.array([]))
        elif isinstance(array, np.ndarray):
            array = np.append(array, np.NAN)
        else:
            raise ValueError("append_to_array is implemented only for list and np.array.")
        res.append(array)
    return res