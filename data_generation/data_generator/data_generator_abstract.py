from abc import ABC, abstractmethod
import numpy as np
from data_generation.model_parameter import Model_parameter
import data_generation.model_parameter
import data_generation.numerical_method.finite_difference.payoff


def cost(l: int, N: int, numerical_method_L: list):
    """
    Returns the computational cost for level l.
    """
    numerical_method = numerical_method_L[l]
    cost = N * numerical_method.cost() # add cost fine grid
    if l > 0:
        numerical_method_c = numerical_method_L[l - 1]
        cost += N * numerical_method_c.cost() # add cost coarse grid
    return cost


class Data_generator_abstract(ABC):
    """
    Abstract class for the data generation.
    """
    def __init__(self, parameter_tuple: tuple, payoff_tuple: tuple, how: str):
        assert len(parameter_tuple) == 2 and len(payoff_tuple) == 2
        # complicated name for reference to Jupyter notebook
        assert all([isinstance(param, data_generation.model_parameter.Model_parameter) for param in parameter_tuple])
        assert all([isinstance(payoff, data_generation.numerical_method.finite_difference.payoff.Payoff) for payoff in payoff_tuple])
        param_min, param_max = parameter_tuple
        assert all([el_min <= el_max for el_min, el_max in zip(param_min, param_max)]), \
               "Parameter tuple does not have min/ max arguments in index 0/ 1, respectively."
        self.parameter_tuple = parameter_tuple
        self.payoff_tuple = payoff_tuple
        assert type(self.parameter_tuple[0]) == type(self.parameter_tuple[1]), "Parameter tuple not of the same type"
        assert type(self.payoff_tuple[0]) == type(self.payoff_tuple[1]), "Payoff tuple not of the same type"
        self.parameter_class = type(self.parameter_tuple[0])
        self.payoff_class = type(self.payoff_tuple[0])
        assert how in {"random", "grid"}, "type can only be random or grid"
        self.how = how

    @abstractmethod
    def test(self, N: int) -> tuple :
        """
        Generates M testing data
        """
        pass

    def get_nbr_payoff_parameters(self):
        return len(self.payoff_tuple[0].get_param())

    def get_low_high_tuple(self, drop_constant_parameter: bool):
        """
        Returns a tuple of torch.DoubleTensor
        """
        parameter_min, parameter_max = self.parameter_tuple
        payoff_min, payoff_max = self.payoff_tuple
        low = np.array([*parameter_min, *payoff_min.get_param()])
        high = np.array([*parameter_max, *payoff_max.get_param()])
        if drop_constant_parameter :
            is_varying = low != high
            low, high = low[is_varying], high[is_varying]
        return low, high

    def get_dim(self, drop_constant_parameter):
        return len(self.get_low_high_tuple(drop_constant_parameter=drop_constant_parameter)[0])

    def drop_constant_parameter(self, x: np.ndarray):
        """
        Drops constant parameters from x.
        """
        low, high = self.get_low_high_tuple(drop_constant_parameter=False)
        return x[:, high - low != 0]

    def get_grid(self,
                 N: int,
                 drop_constant_parameter: bool):
        """
        Returns the points defining the regular grid for the RegularGridInterpolator function.
        The points in each dimension (i.e. every element of the points tuple) is strictly ascending or descending.
        """
        if self.how != "grid":
            raise ValueError("Data must be generated on a grid to get the grid.")
        low, high = self.get_low_high_tuple(drop_constant_parameter=drop_constant_parameter)
        N_1_dim = np.where(high - low != 0, round(N ** (1 / self.get_dim(drop_constant_parameter=True))), 1)
        assert np.isclose(N_1_dim,
                          np.where(high - low != 0, N ** (1 / self.get_dim(drop_constant_parameter=True)), 1),
                          atol=1e-15).all(), (
            "{0} does not define a regular grid on the parameter space with dimension {1}".format(
                N, self.get_dim(drop_constant_parameter=True)))
        grid = tuple(np.linspace(start=l, stop=h, endpoint=True, num=size) for l, h, size in zip(low, high, N_1_dim))
        return grid

    def get_x(self,
              N: int,
              drop_constant_parameter: bool):
        """
        returns the parameters in a numpy array x.
        Does not drop constant parameters.
        """
        if N <= 0:
            raise ValueError("N > 0 needed")
        if self.how == "random":
            low, high = self.get_low_high_tuple(drop_constant_parameter=drop_constant_parameter)
            x = np.random.uniform(low=low, high=high, size=(N, len(low)))
        elif self.how == "grid":
            grid = self.get_grid(N=N, drop_constant_parameter=drop_constant_parameter)
            xg = np.array(np.meshgrid(*grid, indexing="ij", sparse=False))
            x = xg.reshape(self.get_dim(drop_constant_parameter=drop_constant_parameter), -1).T
        else:
            raise ValueError("type of sampling not recognized.")
        return x

    def parameter(self,
                  x: np.ndarray):
        """
        Generates N parameters.
        It is faster by a factor of ca. 1.5 if x includes the constant parameters, but it is not required.
        Input:
        x                       -- np.ndarray of shape (N, self.get_dim(drop_constant_parameter=b)), b is True or False
        Output:
        model_parameter_arr     -- np.ndarray of samples of the parameter class returned by get_parameter class,
                                    e.g. Heston_parameter(S_0, V_0, r, kappa, theta, omega, rho)
        payoff_arr              -- np.ndarray of samples of the payoff class returned by get_payoff_class,
                                    e.g. Call(T,K)
        The latter two outputs are needed for the method numerical_method.evaluate_multiple()
        todo: improve efficiency of the methods apply_along_axis
        """
        if x.shape[1] == self.get_dim(drop_constant_parameter=False):
            model_parameter_arr = x[:, :-self.get_nbr_payoff_parameters()]
            model_parameter_arr = np.apply_along_axis(lambda r: self.parameter_class(*r), 1, model_parameter_arr)
            payoff_parameter_arr = x[:, -self.get_nbr_payoff_parameters():]
            payoff_arr = np.apply_along_axis(lambda r: self.payoff_class(*r), 1, payoff_parameter_arr)
        elif x.shape[1] == self.get_dim(drop_constant_parameter=True):
            low, high = self.get_low_high_tuple(drop_constant_parameter=False)
            b = high - low != 0 # entries that vary
            constant = low[~b]  # constant entries
            b_model = b[:-self.get_nbr_payoff_parameters()]
            idx_cut = b_model.sum()  # index for cutting x
            model_parameter_arr = x[:, :idx_cut]
            def f_model(r) :
                v = np.empty(len(b_model))
                v[b_model] = r
                v[~b_model] = constant[:(~b_model).sum()]
                return self.parameter_class(*v)
            model_parameter_arr = np.apply_along_axis(f_model, 1, model_parameter_arr)

            b_payoff = b[-self.get_nbr_payoff_parameters() :]
            payoff_parameter_arr = x[:, idx_cut :]
            def f_payoff(r) :
                v = np.empty(len(b_payoff))
                v[b_payoff] = r
                v[~b_payoff] = constant[(~b_model).sum():]
                return self.payoff_class(*v)
            payoff_arr = np.apply_along_axis(f_payoff, 1, payoff_parameter_arr)
        else:
            raise ValueError("x has wrong shape")
        return model_parameter_arr, payoff_arr

    def err_numerical_approximation_from_parameters(self,
                                                    l: int,
                                                    x: np.ndarray,
                                                    numerical_method_L: list):
        """
        Generates samples with parameter x for level l according
        to numerical_method_L[l] and calculates the corresponding
        computational cost.
        Input:
        l:                  level
        x:                  np.ndarray of shape (N, self.get_dim(drop_constant_parameter=b)), b is True or False
        numerical_method_L: list of numerical methods
        Output:
        """
        N = int(x.shape[0])  # N is long-int if needed to avoid cost overflow
        numerical_method = numerical_method_L[l]
        model_parameter_arr, payoff_arr = self.parameter(x)
        print("l:", "{:<1}".format(l),
              "nbr samples:", "{:<7}".format(N),
              "dim:", "{:<2}".format(self.get_dim(drop_constant_parameter=True)),
              "numerical method:", numerical_method,
              flush=True)
        y = numerical_method.evaluate_multiple(model_parameter_arr, payoff_arr)
        if l == 0:
            err = y
        else :
            numerical_method_c = numerical_method_L[l - 1]
            y_c = numerical_method_c.evaluate_multiple(model_parameter_arr, payoff_arr)
            err = y - y_c
        return y, err

    def err_numerical_approximation(self,
                                    l: int,
                                    N: int,
                                    numerical_method_L: list):
        """
        Generates N samples for level l according to numerical_method_L[l]
        and calculates the corresponding computational cost.
        """
        x = self.get_x(N, drop_constant_parameter=False)
        y, err = self.err_numerical_approximation_from_parameters(l, x, numerical_method_L)
        x = self.drop_constant_parameter(x)
        cst = cost(l=l, N=N, numerical_method_L=numerical_method_L)
        return x, y, err, cst