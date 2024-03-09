import numpy as np
from dataclasses import dataclass, astuple


@dataclass
class Model_parameter:
    """place-holder class"""
    def __iter__(self):
        """
        Enables to unpack model-parameters, e.g.
            S_0, sigma, r = bs_parameter
            S_0, V_0, r, kappa, theta, omega, rho = heston_parameter
        """
        return iter(astuple(self))


@dataclass
class Bs_parameter(Model_parameter):
    """
    Data-class that stores the parameters of the Black-Scholes model.
    """
    S_0: float
    sigma: float
    r: float

    def __repr__(self):
        return "S_0: " + str(np.round(self.S_0, 2)) + \
               " sigma: " + str(np.round(self.sigma, 2)) + \
               " r: " + str(np.round(self.r, 2))


@dataclass
class Heston_parameter(Model_parameter):
    """
    Data-class that stores the parameters of the Heston model.
    """
    S_0: float
    V_0: float
    r: float
    kappa: float
    theta: float
    omega: float
    rho: float

    def __repr__(self):
        return "S_0: " + str(np.round(self.S_0, 2)) + \
               " V_0: " + str(np.round(self.V_0, 2)) + \
               " r: " + str(np.round(self.r, 2)) + \
               " kappa: " + str(np.round(self.kappa, 2)) + \
               " theta: " + str(np.round(self.theta, 2)) + \
               " omega: " + str(np.round(self.omega, 2)) + \
               " rho: " + str(np.round(self.rho, 2))
