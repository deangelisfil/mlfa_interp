import numpy as np
from scipy.interpolate import RegularGridInterpolator
from function_approximation.multilevel_model import Multilevel_model
from abc import ABC, abstractmethod
from dataclasses import dataclass
import concurrent.futures
from auxiliary_function.error import maxe
import matlab.engine

def tensor_grid_interp(grid: tuple,
                       y: np.ndarray,
                       x_val: np.ndarray = None,
                       y_val: np.ndarray = None):
    """
    grid:       tuple of ndarray of float, with shapes (m_1,), ..., (m_dim,)
    y:          array_like, shape (N,)
    x_val:      array_like, shape (N_val, dim)
    y_val:      array_like, shape (N_val,)
    """
    assert (x_val is None and y_val is None) or (x_val is not None and y_val is not None)
    dim = len(grid)
    assert all([len(axis)==len(grid[0]) for axis in grid]), "The grid must be a tensor grid."
    N_1_dir = len(grid[0])
    assert y.shape == (N_1_dir**dim,1), "The shape of y is not correct."
    y = y.reshape(*((N_1_dir,) * dim))
    interp = RegularGridInterpolator(points=grid, values=y)
    if x_val is None and y_val is None:
        return interp
    else:
        assert len(x_val) == len(y_val)
        interp_err = maxe(y_val - interp(x_val)[:, np.newaxis])
        print("Number of samples:", "{:<6}".format(y.size), "Maximum error:", "{:.3e}".format(interp_err), flush=True)
        return interp_err


def sparse_grid_interp(eng: matlab.engine):
    """
    Input engine:
    - k
    - z_train: resulting z from the training data
    - x: mat with the x entries in columns
    - err: vector with the y entries
    - grid_m1:
    - err_m1:
    """
    if eng.eval("k", nargout=1) == 0 and eng.eval("options.GridType;", nargout=1) == "Maximum":
        # use tensor_grid_interp
        x_val = np.array(eng.eval("x", nargout=1))
        y_val = np.array(eng.eval("err", nargout=1))
        # grid = np.array(eng.eval("grid_m1", nargout=1))
        grid = tuple(np.array(array).ravel() for array in eng.eval("grid_m1", nargout=1))
        y = np.array(eng.eval("err_m1", nargout=1))
        interp_err = tensor_grid_interp(grid=grid, y=y, x_val=x_val, y_val=y_val)
    else:
        # use spinterp
        eng.eval("err_interp = spinterp(z_train, x);", nargout=0)
        interp_err = eng.eval("max(abs(err_interp - err));", nargout=1)
        print("Number of samples:", "{:<6}".format(eng.eval("z_train.nPoints;", nargout=1)),
              "Maximum error:", "{:.3e}".format(interp_err),
              flush=True)
    return interp_err


def generate_fit_multilevel(eng: matlab.engine = None,
                            get_grid: callable = None,
                            grid_interp: callable = None):
    """
    Generates a multilevel fitting function based on interpolation over a grid.

    Args:
    eng:            MATLAB engine instance for spinterp.
    get_grid:       Callable that generates a tensor grid.
                    The interface of grid_interp is
                        interp = grid_interp(grid, y)
    grid_interp:    Callable that performs tensor grid interpolation over a grid.
                    The interface of get_grid is
                        grid = get_grid(N, drop_constant_parameters)
    Returns:
    A multilevel interpolator.
    """

    if eng is None :
        # tensor grid interpolation
        def fit_multilevel(xl: list,
                           errl: list,
                           x_test: np.ndarray,
                           y_test: np.ndarray) -> float:
            """
            Returns the maximum error of the multilevel interpolation compared to test data.

            Args:
                xl: List of np.ndarray with x values of numerical approximations on each level.
                errl: List of np.ndarray with numerical approximations on each level.
                x_test: Test data x values.
                y_test: Test data y values.

            Returns:
                The maximum absolute error between the interpolated values and the test data.
            """
            # use low-level routine get_grid and grid_interp
            y = np.zeros(len(y_test))
            for x, err in zip(xl, errl) :
                grid = get_grid(N=len(err), drop_constant_parameter=True)
                interp = grid_interp(grid=grid, y=err)
                y += interp(x_test)
            return maxe(y[:, np.newaxis] - y_test)
    else:
        # sparse grid interpolation
        def fit_multilevel(x_test: np.ndarray,
                           y_test: np.ndarray) -> float:
            # Assuming prior initialization of MATLAB engine and workspace
            # Note: k[l] >= 1 for every l. So, spinterp is used for every l.
            eng.workspace["x_test"] = x_test
            eng.workspace["y_test"] = y_test
            eng.eval("y = zeros(length(y_test), 1);", nargout=0)
            L = int(eng.eval("sum(~cellfun(@isempty, z));", nargout=1)) # count number of non-empty entries in z
            for l in range(L) :
                eng.workspace["l"] = l
                eng.eval("y = y + spinterp(z{l+1}, x_test);", nargout=0)
            return eng.eval("max(abs(y - y_test));", nargout=1)

    return fit_multilevel