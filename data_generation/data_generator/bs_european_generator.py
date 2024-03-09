from data_generation.model_parameter import Bs_parameter
from data_generation.numerical_method.finite_difference.payoff import Call, Put, Digital_call, Dirac, Digital_put
from auxiliary_function.bs_european import bs_call, bs_put, bs_digital_call, bs_digital_put, bs_dirac
import data_generation
from data_generation.data_generator.data_generator_abstract import Data_generator_abstract


class Bs_european_generator(Data_generator_abstract) :
    """
    Implements the Black Scholes European call and digital call generator.
    """
    def __init__(self, bs_parameter_tuple: tuple, european_tuple: tuple, how: str) :
        super().__init__(bs_parameter_tuple, european_tuple, how)
        assert all([isinstance(param, Bs_parameter) for param in bs_parameter_tuple])
        assert (all([isinstance(call, Call) for call in european_tuple]) or
                all([isinstance(put, Put) for put in european_tuple]) or
                all([isinstance(digital_call, Digital_call) for digital_call in european_tuple]) or
                all([isinstance(digital_put, Digital_put) for digital_put in european_tuple]) or
                all([isinstance(dirac, Dirac) for dirac in european_tuple]))
        if isinstance(european_tuple[0], Call):
            self.opt = "call"
        elif isinstance(european_tuple[0], Put):
            self.opt = "put"
        elif isinstance(european_tuple[0], Digital_call):
            self.opt = "digital call"
        elif isinstance(european_tuple[0], Digital_put):
            self.opt = "digital put"
        elif isinstance(european_tuple[0], Dirac):
            self.opt = "dirac"
        else:
            raise ValueError("the payoff is none of the European options.")

    def test(self, N: int, how = "random") -> tuple:
        """
        Per default the parameters are randomly sampled.
        """
        assert how == "random" if self.how == "random" else True
        if how == self.how:
            x_test = self.get_x(N, drop_constant_parameter=False)
        else:
            tmp = self.how
            self.how = how
            x_test = self.get_x(N, drop_constant_parameter=False)
            self.how = tmp # set back to original value
        if self.opt == "call":
            y_test = bs_call(*x_test.T, opt="value")
        elif self.opt == "put":
            y_test = bs_put(*x_test.T, opt="value")
        elif self.opt == "digital call":
            y_test = bs_digital_call(*x_test.T, opt="value")
        elif self.opt == "digital put":
            y_test = bs_digital_put(*x_test.T, opt="value")
        elif self.opt == "dirac":
            y_test = bs_dirac(*x_test.T)
        else:
            raise ValueError("the payoff is none of the European options.")
        x_test = self.drop_constant_parameter(x_test)
        return x_test, y_test.reshape(-1, 1)
