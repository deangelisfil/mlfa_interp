from data_generation.numerical_method.finite_difference.grid import Grid_2dim, Rustic_grid_2dim
from scipy import sparse
from scipy.sparse.linalg import splu
import numpy as np
from data_generation.numerical_method.finite_difference.payoff import Payoff
from data_generation.numerical_method.numerical_method_abstract import Numerical_method
from data_generation.model_parameter import Bs_parameter

class CflFailure(Exception):
    pass


def bs_matrix(grid, sigma, r) :
    """
    computes the bs_matrix based on grid.S, sigma, r, grid.dt and grid.dS
    M implemented it on purpose with grid instead of grid.S, grid.dt and grid.dS as input parameters
    to ensure that these are choosing according to the same grid. 
    """
    diff = grid.dt / grid.dS
    diff2 = grid.dt / grid.dS ** 2
    a = grid.S ** 2 * sigma ** 2 * diff2 / 2
    b = grid.S * r * diff / 2
    low = - a[1 :] + b[1 :]
    centre = r * grid.dt + 2 * a
    up = - a[:-1] - b[:-1]
    # boundary conditions s.t. d^u/dx^2 = 0
    low[-1] = 2 * b[-1]
    centre[-1] = r * grid.dt - 2 * b[-1]
    B = sparse.diags([low, centre, up], offsets=[-1, 0, 1], format="csc")
    return B


def bs_matrix_multiple(S, sigma, r, dt, dS) :
    """
    implements the calculation of a big bs_matrix based on M input paramters: grid (S, dt, dS), sigma and r
    input:
    S     -> underlying grid, is a numpy.ndarray of dimension (grid_points, M)
    sigma -> volatility, is a numpy.array of dimension M for constant volatility
             and a numpy.ndarray of dimension (grid_points, M) for local volatility
    r     -> interest rate, is a numpy.array of dimension M for constant interest rate
    dt    -> time steps in the underlying grid, is a numpy.array of dimension M
    dS    -> spacial steps in the underlying grid, is a numpy.array of dimension M
    It was slower to implement grid_list, sigma, r instead (as for bs_matrix)
    """
    diff = dt / dS
    diff2 = dt / dS ** 2
    a = S ** 2 * sigma ** 2 * diff2 / 2
    b = S * r * diff / 2
    low = - a[1 :, :] + b[1 :, :]
    centre = r * dt + 2 * a
    up = - a[:-1, :] - b[:-1, :]
    # boundary conditions s.t. d^u/dx^2 = 0
    low[-1, :] = 2 * b[-1, :]
    centre[-1, :] = r * dt - 2 * b[-1, :]
    # create three big arrays by adding a 0 (only) between columns of low and up, and flattening by column
    low = np.vstack((low, np.zeros(low.shape[1]))).flatten("F")[:-1]  # last 0 is not between columns
    centre = centre.flatten("F")
    up = np.vstack((up, np.zeros(up.shape[1]))).flatten("F")[:-1]  # last 0 is not between columns
    # create sparse matrix
    B = sparse.diags([low, centre, up], offsets=[-1, 0, 1], format="csc")
    return B


class Bs_theta(Numerical_method) :
    def __init__(self,
                 theta: float,
                 rustic_grid: Rustic_grid_2dim,
                 execution: str,
                 how: str,
                 rn: bool = False) :
        """
        Class that implements the theta-scheme for the Black-Scholes equation.
        theta is the proportion of implicit scheme.
        how determines how the terminal condition is discretized.
        rn determines whether to use Rannacher startup with 4 fully implicit half-timesteps.
        """
        assert 0 <= theta <= 1
        assert execution in {"european", "american"}
        assert isinstance(rustic_grid, Rustic_grid_2dim)
        assert rn is False if theta != 0.5 else True
        super().__init__(rustic_grid=rustic_grid, how=how)
        self.theta = theta
        self.execution = execution
        self.rn = rn

    def __repr__(self) :
        return "BS scheme with theta=" + str(self.theta) + \
               ", execution=" + self.execution + \
               ", discretization=" + self.how + \
               ", Rannacher start-up=" + str(self.rn) + \
               ", and " + str(self.rustic_grid)

    def __eq__(self, other) :
        assert isinstance(other, Bs_theta)
        return self.theta == other.theta and \
               self.rustic_grid == other.rustic_grid and \
               self.execution == other.execution and \
               self.how == other.how and \
               self.rn == other.rn

    def __hash__(self) :
        return hash((self.theta, self.rustic_grid, self.execution, self.how, self.rn))

    def cost(self) :
        assert isinstance(self.rustic_grid, Rustic_grid_2dim)
        cost = self.rustic_grid.cost()
        if self.rn:
            # each of the first two timesteps solves an additional system of equations
            cost += 2 * int(self.rustic_grid.get_J())
        return cost

    def check_cfl(self, grid, bs_parameter):
        if self.theta < 0.5:
            var_per_spacing_max = (grid.S_max * bs_parameter.sigma / grid.dS)**2
            mu_per_spacing_max = grid.S_max * bs_parameter.r / grid.dS
            fac = var_per_spacing_max + bs_parameter.r + np.sqrt(var_per_spacing_max**2 - mu_per_spacing_max**2)
            if (1 - 2 * self.theta) * grid.dt * fac > 2:
                raise CflFailure("CFL condition does not hold with " \
                                 "theta={0}, dt={1}, dS={2}, sigma={3}, r={4} and S_max={5}".format(
                    self.theta, grid.dt, grid.dS, bs_parameter.sigma, bs_parameter.r, grid.S_max))

    def solve_fde(self, B, f) :
        d = B.shape[0]
        B_exp = sparse.identity(d, format="csc") - (1 - self.theta) * B
        B_imp = sparse.identity(d, format="csc") + self.theta * B
        u = np.copy(f)
        superlu = splu(B_imp)
        for m in reversed(range(self.rustic_grid.M)) :
            if self.rn and m < 2 :
                # two fully implicit half-timesteps
                u = superlu.solve(u)
                if self.execution == "american" :
                    u = np.maximum(u, f)
                u = superlu.solve(u)
            else :
                u = B_exp.dot(u)
                u = superlu.solve(u)
            if self.execution == "american" :
                u = np.maximum(u, f)
        return u

    def evaluate(self,
                 bs_parameter: Bs_parameter,
                 payoff: Payoff,
                 output="u_0") :
        """
        output="u_0" -- outputs the price at time 0 for bs_parameter.S_0
        output="u"   -- outputs the whole price vector u at time 0
        """
        assert output in {"u_0", "u"}
        grid = Grid_2dim(self.rustic_grid, bs_parameter.S_0, payoff.T)
        self.check_cfl(grid, bs_parameter)

        B = bs_matrix(grid, bs_parameter.sigma, bs_parameter.r)
        f = payoff.discretize(grid.S, how=self.how)
        u = self.solve_fde(B=B, f=f)  # solve sparse system of linear equations
        if output == "u" :
            return u
        else :
            assert (self.rustic_grid.get_J() / self.rustic_grid.S_max_ratio).is_integer(), "S_0 is not on the grid."
            j_0 = int(self.rustic_grid.get_J() / self.rustic_grid.S_max_ratio)
            return u[j_0]

    def evaluate_multiple(self,
                          bs_parameter_arr: np.ndarray,
                          payoff_arr: np.ndarray,
                          output="u_0") :
        """
        implements the evaluation of multiple parameters by avoiding overhead (the for-loop)
        to validate this function M can compare the output to the same method implemented in the abstract class.
        """
        # Choose the batch_size such that it is contained in the L2 cache of 1024KB
        # 1024KB = 128,000 doubles, 1 double = 8 bytes
        # numbers stored in B = batch_size*(J+1) + 2*(batch_size*(J+1)-1), roughly B = 3*batch_size*(J+1)
        # So, choose batch_size such that
        # 3*batch_size*(J+1) < 128,000; or to be conservative 3*batch_size*(J+1) < 50,000
        # batch_size = max(int(5e4 / (3 * (self.rustic_grid.get_J() + 1))), 1)
        batch_size = max(int(12.8e4 / (4 * (self.rustic_grid.get_J()))), 1)
        N = len(bs_parameter_arr)
        if output == "u_0" :
            assert (self.rustic_grid.get_J() / self.rustic_grid.S_max_ratio).is_integer(), "S_0 is not on the grid."
            y = np.empty(N, dtype=np.float64)
        elif output == "u" :
            y = np.zeros((N, self.rustic_grid.get_J() + 1), dtype=np.float64)
        else :
            raise ValueError("output value is invalid.")
        for idx in range(0, N, batch_size) :
            bs_parameter_batch = bs_parameter_arr[idx : idx + batch_size]
            payoff_batch = payoff_arr[idx : idx + batch_size]

            # Read parameters in: bs_parameter_arr and payoff_arr
            S_batch = []
            r_batch = []
            sigma_batch = []
            dt_batch = []
            ds_batch = []
            f_batch = []
            # S_0_batch = []
            for bs_parameter, payoff in zip(bs_parameter_batch, payoff_batch) :
                grid = Grid_2dim(self.rustic_grid, bs_parameter.S_0, payoff.T)
                self.check_cfl(grid, bs_parameter)

                f = payoff.discretize(grid.S, how=self.how)
                S_batch.append(grid.S)
                r_batch.append(bs_parameter.r)
                sigma_batch.append(bs_parameter.sigma)
                dt_batch.append(grid.dt)
                ds_batch.append(grid.dS)
                f_batch.append(f)
            S_batch = np.array(S_batch)
            r_batch = np.array(r_batch)
            sigma_batch = np.array(sigma_batch)
            dt_batch = np.array(dt_batch)
            ds_batch = np.array(ds_batch)
            f_batch = np.array(f_batch).flatten()

            # construct big B_batch
            B = bs_matrix_multiple(S_batch.transpose(), sigma_batch, r_batch, dt_batch, ds_batch)
            d_batch = S_batch.shape[0]

            # solve sparse system of linear equations
            u = self.solve_fde(B=B, f=f_batch)
            u = u.reshape(S_batch.shape)
            if output == "u_0" :
                y[idx : idx + d_batch] = u[:, int(self.rustic_grid.get_J() / self.rustic_grid.S_max_ratio)]
            elif output == "u" :
                y[idx : idx + d_batch, :] = u

        if output == "u_0" :
            return y[:, np.newaxis]
        elif output == "u" :
            return y
