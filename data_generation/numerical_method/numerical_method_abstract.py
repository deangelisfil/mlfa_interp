from abc import ABC, abstractmethod
from data_generation.numerical_method.finite_difference.payoff import Payoff
from data_generation.model_parameter import Model_parameter
import numpy as np
import copy


class Numerical_method(ABC) :
    def __init__(self,
                 rustic_grid,
                 how: str) :
        assert how in {"ptw", "simple", "hat"}
        self.rustic_grid = copy.deepcopy(rustic_grid)  # otherwise it is passed by reference
        self.how = how

    def create_list_in_time(self, M_L) :
        """
        creates a list of numerical_method with grid with M timesteps in M_L.
        Ignores self.rustic_grid.M as it chooses M in M_L.
        """
        numerical_method_list = []
        for M in M_L :
            numerical_method = copy.deepcopy(self)  # uses the default implementation of __copy__
            numerical_method.rustic_grid.M = M
            numerical_method_list.append(numerical_method)
        return numerical_method_list

    def create_list_in_space(self, J_L):
        assert self.rustic_grid.grid_type == "fixed", "can create list in space only if the rustic grid has fixed J."
        numerical_method_list = []
        for J in J_L:
            numerical_method = copy.deepcopy(self)
            numerical_method.rustic_grid.J_fixed = J
            numerical_method_list.append(numerical_method)
        return numerical_method_list

    @abstractmethod
    def cost(self):
        pass

    @abstractmethod
    def evaluate(self, model_parameter: Model_parameter, payoff: Payoff) :
        pass

    def evaluate_multiple(self, model_parameter_arr: np.array, payoff_arr: np.array) :
        """
        output y is of the type torch.float64 (because of precision needs).
        slow implementation with for loop, see specific class implementation to avoid overhead issues.
        """
        assert len(model_parameter_arr) == len(payoff_arr)
        N = len(model_parameter_arr)
        y = np.zeros((N, 1))
        for i in range(N):
            y[i] = self.evaluate(model_parameter_arr[i], payoff_arr[i])
        return y


