from data_generation.model_parameter import Bs_parameter
from data_generation.numerical_method.finite_difference.payoff import Call
from auxiliary_function.heston import heston_call
import data_generation
import torch
from data_generation.data_generator.data_generator_abstract import Data_generator_abstract
import numpy as np

class Heston_call_generator(Data_generator_abstract):
    def __init__(self, heston_parameter_tuple: tuple, call_tuple: tuple, how: str):
        super().__init__(heston_parameter_tuple, call_tuple, how)
        assert all([isinstance(param, data_generation.model_parameter.Heston_parameter) for param in heston_parameter_tuple])
        assert all([isinstance(call, data_generation.numerical_method.finite_difference.payoff.Call) for call in call_tuple])

    def test(self, N: int, how = "random") -> tuple:
        assert how == "random" if self.how == "random" else True
        if how == self.how:
            x_test = self.get_x(N, drop_constant_parameter=False)
        else:
            tmp = self.how
            self.how = how
            x_test = self.get_x(N, drop_constant_parameter=False)
            self.how = tmp # set back to original value
        y_test = np.zeros((N, 1), dtype=np.float64)
        for i, x in enumerate(x_test):
            y_test[i] = heston_call(*x)
        x_test = self.drop_constant_parameter(x_test)
        return x_test, y_test