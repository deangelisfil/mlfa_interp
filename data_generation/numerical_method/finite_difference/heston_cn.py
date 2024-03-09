import numpy as np
from scipy import sparse
from scipy.sparse import spdiags
from scipy.sparse.linalg import splu
from data_generation.numerical_method.numerical_method_abstract import Numerical_method
from data_generation.model_parameter import Heston_parameter
from data_generation.numerical_method.finite_difference.payoff import Payoff
from data_generation.numerical_method.finite_difference.grid import Rustic_grid_3dim, Grid_3dim
import copy
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


def heston_cn(grid: Grid_3dim,
              r: float,
              kappa: float,
              theta: float,
              omega: float,
              rho: float,
              V: np.ndarray,
              option: str) :
    """
    Finite difference discretization of the PDE resulting from the Heston model
    using Crank-Nicholson finite difference scheme.
    I follow the Matlab code of Mike Giles in heston_df.M.
    Please look at the above mentioned code for the adjoints,
    they are not included in this version of the code.
    Inputs:
    grid   -- 3 dimensional uniform grid, includes:
                Smax    -- extent of computational grid in S direction
                volmax  -- extent of computational grid in V direction
                dt      -- delta in time
                dS      -- delta in S
                dV      -- delta in V
    r      -- interest rate
    kappa  -- mean reversion rate
    theta  -- mean reversion volatility
    omega  -- vol-of-variance coefficient
    rho    -- correlation factor
    V      -- value at maturity on uniform grid (corresponding to "initial data")
    option -- 'CN1' Crank-Nicolson without Craig-Sneyd correction
           -- 'CN2' Crank-Nicolson with Craig-Sneyd correction
           -- 'RT1' like CN2 but with Rannacher 1 startup
           -- 'RT2' like CN2 but with Rannacher 2 startup
    Outputs:
    V      -- Value at time 0
    """

    #
    # check that theta is not too big for the variance grid
    #

    assert 2 * theta <= grid.V_max, "theta={0} too big for the variance grid [0, {1}]".format(theta, grid.V_max)

    #
    # construct matrices
    #

    J = grid.rustic_grid.get_J() + 1  # attention: we only use this modified definition of J and K to match Matlab code of Mike
    K = grid.rustic_grid.get_K() + 1
    JK = J * K

    con1 = 0.5 / grid.dS ** 2
    con2 = 0.5 / grid.dS
    u1 = r * grid.x1
    sig1 = np.maximum(grid.x2 * grid.x1 ** 2, u1 * grid.dS)
    sig1[0, :] = u1[0, :] * grid.dS  # zero second derivative
    sig1[-1, :] = -u1[-1, :] * grid.dS  # zero second derivative
    u1 = np.reshape(u1, JK, "F")  # F maintains Matlab order
    sig1 = np.reshape(sig1, JK, "F")  # F maintains Matlab order
    low = con1 * sig1 - con2 * u1
    centre = -2 * con1 * sig1 - r
    up = con1 * sig1 + con2 * u1
    D1 = sparse.spdiags([up, centre, low], [-1, 0, 1], JK, JK, format="csr").T
    # Note: swap of low-up on purpose since the matrix is then transposed.
    #       it is also on purpose that format is "csr" since transposing will make it "csc", what we are interested in

    con1 = 0.5 / grid.dV ** 2
    con2 = 0.5 / grid.dV
    u2 = kappa * (theta - grid.x2)
    sig2 = np.maximum(omega ** 2 * grid.x2, np.abs(u2 * grid.dV))
    sig2[:, 0] = u2[:, 0] * grid.dV  # zero second derivative
    sig2[:, -1] = -u2[:, -1] * grid.dV  # zero second derivative
    u2 = np.reshape(u2, JK, "F")
    sig2 = np.reshape(sig2, JK, "F")

    low = con1 * sig2 - con2 * u2
    centre = -2 * con1 * sig2
    up = con1 * sig2 + con2 * u2
    D2 = spdiags([up, centre, low], [-J, 0, J], JK, JK, format="csr").T
    # Note: swap of low-up on purpose since the matrix is then transposed.
    #       it is also on purpose that format is "csr" since transposing will make it "csc", what we are interested in

    con1 = 0.5 / (grid.dS * grid.dV)
    fac = np.vstack((np.zeros((1, K)),
                     np.hstack((np.zeros((J - 2, 1)), np.ones((J - 2, K - 2)), np.zeros((J - 2, 1)))),
                     np.zeros((1, K))))
    sig12 = np.reshape(rho * omega * grid.x2 * grid.x1 * fac, JK, "F")

    if rho > 0 :
        D12 = spdiags([0 * sig12, -con1 * sig12, con1 * sig12,
                       -con1 * sig12, 2 * con1 * sig12, -con1 * sig12,
                       con1 * sig12, -con1 * sig12, 0 * sig12],
                      [-J + 1, -J, -J - 1, 1, 0, -1, J + 1, J, J - 1], JK, JK, format="csr").T
    else :
        D12 = spdiags([-con1 * sig12, con1 * sig12, 0 * sig12,
                       con1 * sig12, -2 * con1 * sig12, con1 * sig12,
                       0 * sig12, con1 * sig12, -con1 * sig12],
                      [-J + 1, -J, -J - 1, 1, 0, -1, J + 1, J, J - 1], JK, JK, format="csr").T

    superlu1 = splu(sparse.identity(JK, format="csc") - 0.5 * grid.dt * D1, diag_pivot_thresh=0)
    superlu2 = splu(sparse.identity(JK, format="csc") - 0.5 * grid.dt * D2, diag_pivot_thresh=0)

    D = grid.dt * (D1 + D2 + D12)
    Dcorr = grid.dt * D12

    #
    # main timemarching loop
    #

    V = np.reshape(V, JK, "F")

    for m in range(grid.rustic_grid.M) :
        if option == 'CN1' :
            V = V + superlu2.solve(superlu1.solve(D * V))
        elif option == 'CN2' :
            corr = 0.5 * Dcorr * superlu2.solve(superlu1.solve(D * V))
            V = V + superlu2.solve(superlu1.solve(D * V + corr))
        elif option == 'RT1' :
            if m < 2 :
                V = V + superlu2.solve(superlu1.solve(0.5 * D * V))
                V = V + superlu2.solve(superlu1.solve(0.5 * D * V))
            else :
                corr = 0.5 * Dcorr * superlu2.solve(superlu1.solve(D * V))
                V = V + superlu2.solve(superlu1.solve(D * V + corr))
        elif option == 'RT2' :
            if m < 2 :
                V = V + 0.5 * Dcorr * V
                V = superlu2.solve(superlu1.solve(V))
                V = V + 0.5 * Dcorr * V
                V = superlu2.solve(superlu1.solve(V))
            else :
                corr = 0.5 * Dcorr * (superlu2.solve(superlu1.solve(D * V)))
                V = V + superlu2.solve(superlu1.solve(D * V + corr))
        else :
            raise ValueError("invalid option value")

    return np.reshape(V, (J, K), "F")


class Heston_cn(Numerical_method):
    def __init__(self,
                 rustic_grid: Rustic_grid_3dim,
                 how: str) :
        assert isinstance(rustic_grid, Rustic_grid_3dim)
        super().__init__(rustic_grid=rustic_grid, how=how)

    def __repr__(self):
        return "Heston Crank-Nicolson scheme with " + str(self.rustic_grid)

    def __hash__(self):
        return hash((self.rustic_grid, self.how))

    def __eq__(self, other):
        assert isinstance(other, Heston_cn)
        return self.rustic_grid == other.rustic_grid and self.how == other.how

    def cost(self):
        cost = self.rustic_grid.cost()
        cost += 2 * int(self.rustic_grid.get_J()) * int(self.rustic_grid.get_K()) # Rannacher startup
        return cost

    def evaluate(self, heston_parameter: Heston_parameter, payoff: Payoff, output="u_0", option="RT2") :
        """
        Uses option RT2
        output="u_0" -- outputs the price at time 0 for bs_parameter.S_0
        output="u"   -- outputs the whole price vector u at time 0
        """
        assert output in {"u_0", "u"}
        grid = Grid_3dim(self.rustic_grid, heston_parameter.S_0, heston_parameter.V_0, payoff.T)
        u_final = payoff.discretize(grid.x1, how=self.how)
        u = heston_cn(grid,
                      heston_parameter.r,
                      heston_parameter.kappa,
                      heston_parameter.theta,
                      heston_parameter.omega,
                      heston_parameter.rho,
                      u_final,
                      option)
        if output == "u":
            return u
        else :
            assert (self.rustic_grid.get_J() / self.rustic_grid.S_max_ratio).is_integer(), "S_0 not on the grid"
            i_0 = int(self.rustic_grid.get_J() / self.rustic_grid.S_max_ratio)
            if (heston_parameter.V_0 == grid.V_max / self.rustic_grid.V_max_ratio
                    and (self.rustic_grid.get_K() / self.rustic_grid.V_max_ratio).is_integer()):
                j_0 = int(self.rustic_grid.get_K() / self.rustic_grid.V_max_ratio)
                u_0 = u[i_0, j_0]
            else:
                assert heston_parameter.V_0 not in grid.V
                # cubic spline interpolation in the v-direction
                # 'not-a-knot spline' for small V and linear/natural condition for big V
                cs = CubicSpline(x=grid.V, y=u[i_0, :], bc_type=("not-a-knot", (2, 0.0)))
                u_0 = cs(heston_parameter.V_0)
            return u_0
