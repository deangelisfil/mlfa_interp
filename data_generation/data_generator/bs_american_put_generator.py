from data_generation.model_parameter import Bs_parameter
from data_generation.numerical_method.finite_difference.payoff import Payoff
import data_generation
from data_generation.data_generator.data_generator_abstract import Data_generator_abstract


class Bs_american_put_generator(Data_generator_abstract) :
    def __init__(self, bs_parameter_tuple: tuple, put_tuple: tuple, how: str) :
        super().__init__(bs_parameter_tuple, put_tuple, how)
        assert all([isinstance(param, data_generation.model_parameter.Bs_parameter) for param in bs_parameter_tuple])
        assert all([isinstance(put, data_generation.numerical_method.finite_difference.payoff.Put) for put in put_tuple])

    def test(self, N: int) -> tuple:
        raise NotImplementedError("No explicit formula for the American put.")