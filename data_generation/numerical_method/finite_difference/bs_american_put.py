import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import splu
from data_generation.numerical_method.numerical_method_abstract import Numerical_method
from data_generation.model_parameter import Heston_parameter, Model_parameter
from data_generation.numerical_method.finite_difference.payoff import Payoff
from data_generation.numerical_method.finite_difference.grid import Grid_2dim
from data_generation.numerical_method.finite_difference.grid import Rustic_grid_2dim, Grid_2dim
import copy
from data_generation.numerical_method.finite_difference.payoff import Put
from data_generation.model_parameter import Bs_parameter
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


def bs_american_put(rustic_grid: Rustic_grid_2dim,
                    S_max: float,
                    sigma: float,
                    r: float,
                    put: Put,
                    atol: float = 1e-8,
                    nmax: float = 100):
    """
    Finite difference discretization of an american option under the Black-Scholes model.
    Uses an implicit discretization with moving grid.
    The calculations involve a Newton iteration at each time step.
    I follow the Matlab code of Mike Giles in amer_3.m.
    Since the grid is moving, the values grid.S and grid.dS are not used.
    The grid Sp ranges from the value of Spb to S_max. So, at maturity it is [K, S_max].
    Vp is the value of the option on the grid. The value for smaller Sp is its exercise value K-S.
    Inputs:
    sigma  -- volatility
    r      -- interest rate
    T      -- maturity
    K      -- strike
    M      -- number of time steps
    J      -- number of space steps
    S_max  -- maximal number of S in the grid
    Outputs:
    Sp     -- vector J+2 with the grid for the stock price.
    Vp     -- vector J+2 with the value of the option corresponding to the grid SS
    """
    J = rustic_grid.get_J()
    M = rustic_grid.M

    eta = np.linspace(start=-1/J, stop=1,  num=J+2, endpoint=True)
    deta = 1 / J
    j = np.arange(1, J+1)
    jm = j - 1
    jp = j + 1

    Sbp = put.K
    Sp = Sbp + eta * (S_max - Sbp)
    dSp = Sp[1] - Sp[0]
    Vp = put.discretize(S=Sp, how="ptw") # for moving grid ptw is enough

    # Time-marching
    for m in range(1, M + 1):
        dt = put.T * (m ** 2 - (m - 1) ** 2) / M ** 2 # non-uniform time step

        if m > 1:
            Sbp2 = 2 * Sbp - Sbm
            Sp2 = 2 * Sp - Sm
            dSp2 = 2 * dSp - dSm
            Vp2 = 2 * Vp - Vm

        Sbm = Sbp
        Sm = copy.deepcopy(Sp)
        dSm = dSp
        Vm = copy.deepcopy(Vp)

        if m > 1:
            Sbp = copy.deepcopy(Sbp2)
            Sp = copy.deepcopy(Sp2)
            dSp = copy.deepcopy(dSp2)
            Vp = copy.deepcopy(Vp2)

        # Newton iteration

        delta_Sb = 1
        delta_V = 1
        kount = 0

        while abs(delta_Sb) > atol and np.linalg.norm(delta_V) > atol and kount < nmax:
            kount += 1

            S = 0.5 * (Sp[j] + Sm[j])
            Sbdot = (Sbp - Sbm) / dt
            Vdot = (Vp[j] - Vm[j]) / dt
            V = 0.5 * (Vp[j] + Vm[j])
            Vs = 0.25 * ((Vp[jp] - Vp[jm]) / dSp + (Vm[jp] - Vm[jm]) / dSm)
            Vss = 0.5 * ((Vp[jp] - 2 * Vp[j] + Vp[jm]) / dSp**2 + (Vm[jp] - 2 * Vm[j] + Vm[jm]) / dSm**2)

            res = Vdot + r * V - ((1 - eta[j]) * Sbdot + r * S) * Vs - 0.5 * sigma ** 2 * S ** 2 * Vss

            res_Vjm = -((1 - eta[j]) * Sbdot + r * S) * (-0.25 / dSp) - 0.5 * sigma ** 2 * S ** 2 * (0.5 / dSp ** 2)
            res_Vj = 1 / dt + 0.5 * r + sigma ** 2 * S ** 2 * (0.5 / dSp ** 2)
            res_Vjp = -((1 - eta[j]) * Sbdot + r * S) * (0.25 / dSp) - 0.5 * sigma ** 2 * S ** 2 * (0.5 / dSp ** 2)
            res_Sb = -(1 - eta[j]) * (1 / dt + 0.5 * r) * Vs \
                     - 0.5 * sigma ** 2 * S * (1 - eta[j]) * Vss \
                     - ((1 - eta[j]) * Sbdot + r * S) * 0.25 * (Vp[jp] - Vp[jm]) * deta / dSp ** 2 \
                     - 0.5 * sigma ** 2 * S ** 2 * (Vp[jp] - 2 * Vp[j] + Vp[jm]) * deta / dSp ** 3
            res_Sb[0] -= res_Vjm[0] * (1 - eta[0])

            A = spdiags([res_Vjp, res_Vj, res_Vjm], [-1, 0, 1], J, J, format="csr").T
            superlu = splu(A)

            del1 = -superlu.solve(res)
            del2 = -superlu.solve(res_Sb)

            delta_Sb = (-del1[0] - (Vp[1] - (put.K - Sbp)) ) / (1 + del2[0])
            delta_V = del1 + delta_Sb * del2

            Sbp += delta_Sb
            Sp = Sbp + eta * (S_max - Sbp)
            dSp = Sp[1] - Sp[0]
            Vp[j] += delta_V
            Vp[0] = put.K - Sp[0]

        if kount == nmax and abs(delta_Sb) > atol and np.linalg.norm(delta_V) > atol:
            raise ValueError("Newton iteration did not converge for input parameters:\n"
                             "rustic_grid={},\n"
                             " S_max={}, sigma={}, r={}, put={}, atol={}, nmax={}".format(
                rustic_grid, S_max, sigma, r, put, atol, nmax))

    return Sp, Vp


class Bs_american_put(Numerical_method):
    def __init__(self,
                 rustic_grid: Rustic_grid_2dim,
                 how: str = "ptw") :
        assert isinstance(rustic_grid, Rustic_grid_2dim)
        super().__init__(rustic_grid=rustic_grid, how=how)


    def __repr__(self):
        return "Black-Scholes American put with " + str(self.rustic_grid)

    def __hash__(self):
        return hash((self.rustic_grid, self.how))

    def __eq__(self, other):
        assert isinstance(other, Bs_american_put)
        return self.rustic_grid == other.rustic_grid and self.how == other.how

    def cost(self):
        return self.rustic_grid.cost()

    def evaluate(self,
                 bs_parameter: Bs_parameter,
                 put: Put,
                 output="u_0") :
        """
        output="u_0" -- outputs the price at time 0 for bs_parameter.S_0
        output="u"   -- outputs the whole price vector u at time 0
        """
        assert output in {"u_0", "u"}
        assert isinstance(put, Put)
        assert bs_parameter.r >= 0, "r must be non-negative"

        S_max = Grid_2dim(rustic_grid=self.rustic_grid, S_0=bs_parameter.S_0, T=put.T).S_max
        S, u = bs_american_put(rustic_grid=self.rustic_grid,
                               S_max=S_max,
                               sigma=bs_parameter.sigma,
                               r=bs_parameter.r,
                               put=put)
        assert np.isclose(put.K - S[0], u[0], atol=1e-14), "S[0] is not the exercise boundary."

        S_prev = np.linspace(start=0, stop=S[0], endpoint=False, num=self.rustic_grid.get_J())
        assert S_prev[1] - S_prev[0] <= S[1] - S[0], "spacing of S_prev is too big."
        S = np.insert(S, 0, S_prev)
        u = np.insert(u, 0, put.K - S_prev)

        if output == "u" :
            return u
        else :
            if bs_parameter.S_0 <= S[0]:
                # exercise value
                assert put.K > bs_parameter.S_0
                return put.K - bs_parameter.S_0, S[0]
            else:
                # continuation value
                # cubic spline interpolation
                # at the exercise boundary the second derivative is 0 and for S >> K the second derivative is also 0.
                cs = CubicSpline(x=S, y=u, bc_type="natural")
                return cs(bs_parameter.S_0)