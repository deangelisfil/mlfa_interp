import numpy as np
from abc import ABC, abstractmethod


class Payoff(ABC) :
    """
    Abstract class representing a payoff function of a European option with maturity T and strike K.
    The payoff is discretized assuming a uniform grid and using the method discretize.
    The discretization method can be pointwise, simple cell averaging and hat-weighted cell averaging.
    """
    def __init__(self, T: float, K: float) -> None:
        self.T = T
        self.K = K
        self.name = "Payoff"

    def __repr__(self) :
        return (self.name
                + " with T=" + str(np.round(self.T, 5))
                + ", K=" + str(np.round(self.K, 5)))

    @abstractmethod
    def discretize(self, S: np.ndarray, how: str) -> np.ndarray:
        pass

    def get_param(self) -> tuple:
        return self.T, self.K


class General_payoff(Payoff):
    """
    General class for options
    """
    def __init__(self, discretize: callable, T: float, K: float) -> None:
        self._discretize = discretize
        self.T = T
        self.K = K

    def discretize(self, S: np.ndarray, how: str) -> np.ndarray:
        if how == "ptw":
            return self._discretize(S)
        else:
            raise ValueError("discretization is only ptw. for general General_payoff instance.")


class Call(Payoff) :
    def __init__(self, T: float, K: float) :
        super().__init__(T, K)
        self.name = "Call"

    def discretize(self, S: np.ndarray, how: str) -> np.ndarray:
        """
        Discretizes the payoff function of a call option with maturity T and strike K on a uniform grid.
        The discretization method is specified by the member self.how.
        :param S: numpy array of stock prices
        :return: numpy array of the discretized payoff
        """
        h = S[1] - S[0]  # assume uniform grid
        res = np.maximum(S - self.K, 0)
        if how == "ptw":
            pass
        elif how == "simple":
            b = np.logical_and(S - h/2 <= self.K, self.K < S + h/2)
            assert np.all(np.sum(b, axis=0) == 1), "K is not in the grid"
            res[b] = 1/(2*h) * (S[b] + h/2 - self.K) ** 2
        elif how == "hat":
            b1 = np.logical_and(S - h <= self.K, self.K < S)
            b2  = np.logical_and(S <= self.K, self.K < S + h)
            assert np.all(np.sum(b1, axis=0) == 1) and np.all(np.sum(b2, axis=0) == 1), "K is not in the grid"
            res[b1] = S[b1] - self.K + (self.K - S[b1] + h)**3 / (6 * h**2)
            res[b2] = (S[b2] + h - self.K)**3 / (6 * h**2)
        else:
            raise ValueError("discretization method is not supported.")
        return res


class Put(Payoff) :
    def __init__(self, T: float, K: float):
        super().__init__(T, K)
        self.name = "Put"

    def discretize(self, S: np.ndarray, how: str) -> np.ndarray:
        h = S[1] - S[0]  # assume uniform grid
        res = np.maximum(self.K - S, 0)
        if how == "ptw":
            pass
        elif how == "simple":
            b = np.logical_and(S - h/2 <= self.K, self.K < S + h/2)
            assert sum(b) == 1, "K is not in the grid"
            res[b] = 1/(2*h) * (self.K - S[b] + h/2) ** 2
        elif how == "hat":
            b1 = np.logical_and(S - h <= self.K, self.K < S)
            b2  = np.logical_and(S <= self.K, self.K < S + h)
            assert np.all(np.sum(b1, axis=0) == 1) and np.all(np.sum(b2, axis=0) == 1), "K is not in the grid"
            res[b1] = (self.K - S[b1] + h)**3 / (6 * h**2)
            res[b2] = self.K - S[b2] + (S[b2] + h - self.K)**3 / (6 * h**2)
        else:
            raise ValueError("discretization method is not supported.")
        return res


class Digital_call(Payoff) :
    def __init__(self, T: float, K: float):
        super().__init__(T, K)
        self.name = "Digital Call"

    def discretize(self, S: np.ndarray, how: str) -> np.ndarray:
        h = S[1] - S[0]
        res = np.heaviside(S - self.K, 1) # take the value 1 at S = K
        if how == "ptw":
            pass
        elif how == "simple":
            b = np.logical_and(S - h/2 <= self.K, self.K < S + h/2)
            assert sum(b) == 1, "K is not in the grid"
            res[b] = (S[b] - self.K) / h + 0.5
        elif how == "hat":
            b1 = np.logical_and(S - h <= self.K, self.K < S)
            b2  = np.logical_and(S <= self.K, self.K < S + h)
            assert sum(b1) == 1 and sum(b2) == 1, "K is not in the grid"
            res[b1] = 1 - (self.K - S[b1] + h)**2 / (2 * h**2)
            res[b2] = (S[b2] + h - self.K)**2 / (2 * h**2)
        else:
            raise ValueError("discretization method is not supported.")
        return res


class Digital_put(Payoff) :
    def __init__(self, T: float, K: float):
        super().__init__(T, K)
        self.name = "Digital Put"

    def discretize(self, S: np.ndarray, how: str) -> np.ndarray:
        h = S[1] - S[0]
        res = np.heaviside(self.K - S, 1) # take the value 1 at S = K
        if how == "ptw":
            pass
        elif how == "simple":
            b = np.logical_and(S - h/2 <= self.K, self.K < S + h/2)
            assert sum(b) == 1, "K is not in the grid"
            res[b] = (self.K - S[b]) / h + 0.5
        elif how == "hat":
            b1 = np.logical_and(S - h <= self.K, self.K < S)
            b2  = np.logical_and(S <= self.K, self.K < S + h)
            assert sum(b1) == 1 and sum(b2) == 1, "K is not in the grid"
            res[b1] = (self.K - S[b1] + h)**2 / (2 * h**2)
            res[b2] = 1 - (S[b2] + h - self.K)**2 / (2 * h**2)
        else:
            raise ValueError("discretization method is not supported.")
        return res


class Dirac(Payoff):
    def __init__(self, T: float, K: float):
        super().__init__(T, K)
        self.name = "Dirac"

    def discretize(self, S: np.ndarray, how: str) -> np.ndarray:
        h = S[1] - S[0]
        res = 1/h * np.isclose(S, self.K, atol=1e-15)
        if how == "ptw":
            pass
        elif how == "simple":
            b = np.logical_and(S - h/2 <= self.K, self.K < S + h/2)
            assert sum(b) == 1, "K is not in the grid"
            res[b] = 1/h
        elif how == "hat":
            b1 = np.logical_and(S - h <= self.K, self.K < S)
            b2  = np.logical_and(S <= self.K, self.K < S + h)
            assert sum(b1) == 1 and sum(b2) == 1, "K is not in the grid"
            res[b1] = (self.K - S[b1]) / h**2 + 1/h
            res[b2] = (S[b2] - self.K) / h**2 + 1/h
        else:
            raise ValueError("discretization method is not supported.")
        return res