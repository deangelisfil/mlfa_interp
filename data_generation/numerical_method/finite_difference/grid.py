import numpy as np


class Rustic_grid_2dim :
    """
    Place-holder class for the construction of the 2-dimensinal grid. This class determines the cost of each sample.
    M           -- number of time steps
    J           -- number of space steps, J depends on M and the type of the grid.
    S_max_ratio -- the ratio with which the class Grid constructs the spatial grid depending on the starting value S_0.
    J_M_ratio   -- the ratio J/M if the grid-type is linear. If the grid-type is non-linear, J_M_ratio is not used.
                   Attention, J_M_ratio is not the same as
                   lambda := dt/dS = J*T / (M*S_max_ratio*S_0) = J_M_ratio * T / (S_max_ratio * S_0)
    """
    def __init__(self, M: int, S_max_ratio: float, grid_type: str, J_M_ratio: int = 16, J_fixed: int = 64) :
        assert grid_type in {"linear", "quadratic", "fixed"}
        assert grid_type == "linear" if J_M_ratio != 16 else True
        assert grid_type == "fixed" if J_fixed != 64 else True
        assert int(J_M_ratio) == J_M_ratio, "J_M_ratio must be an integer"
        self.S_max_ratio = S_max_ratio
        self.grid_type = grid_type
        self.M = M
        self.J_M_ratio = int(J_M_ratio)
        self.J_fixed = int(J_fixed)

    def __repr__(self) :
        ret = "2dim rustic grid with M=" + str(self.M) + \
              ", S_max_ratio=" + str(self.S_max_ratio) + \
              ", grid_type=" + self.grid_type
        if self.grid_type == "linear" :
            ret += ", J_M_ratio=" + str(self.J_M_ratio)
        elif self.grid_type == "fixed" :
            ret += ", J_fixed=" + str(self.J_fixed)
        return ret

    def __eq__(self, other) :
        assert isinstance(other, Rustic_grid_2dim)
        return self.M == other.M and \
               self.S_max_ratio == other.S_max_ratio and \
               self.grid_type == other.grid_type and \
               self.J_M_ratio == other.J_M_ratio and \
               self.J_fixed == other.J_fixed

    def __hash__(self) :
        return hash((self.M, self.S_max_ratio, self.grid_type, self.J_M_ratio))

    def get_J(self) :
        if self.grid_type == "linear" :
            # keeps ratio J/M fixed, i.e. dt/dx for fixed T and S_0
            J = self.J_M_ratio * self.M
        elif self.grid_type == "quadratic" :
            # keep ratio J^2/M fixed, i.e. dt/dx^2 for fixed T and S_0
            # The CFL condition holds roughly (supposing r is not too big and theta=0) for
            # T / M * J**2 * sigma**2 <= 1 which, on the whole parameter space (without T) holds if
            # J <= sqrt(M) / (sqrt(T) * sigma_max).
            # With T = 1 and sigma_max = 0.5 we have exactly 1/(sqrt(T) * sigma_max) = 2.
            assert np.sqrt(self.M).is_integer(), "The number of space steps is not an integer."
            J = 2 * np.sqrt(self.M).astype(int)
        elif self.grid_type == "fixed" :
            J = self.J_fixed
        else :
            raise ValueError("grid type value is invalid.")
        return J

    def cost(self):
        # returns int to avoid cost overflow
        return int(self.M) * int(self.get_J())


class Rustic_grid_3dim :
    """
    Place-holder for the construction of the 3-dimensional grid. This class determines the cost of each sample.
    M -- number of time steps
    J -- number of space steps
    K -- number of volatility steps
    So far, J and K are linear with respect to M.
    """
    def __init__(self,
                 M: int,
                 S_max_ratio: float,
                 V_max_ratio: int,
                 grid_type: str,
                 J_M_ratio: int = 8, # default is 8 for 3dim grid
                 K_M_ratio: float = 0.5,
                 fixed_K: int = None):
        assert grid_type in {"linear", "fixed_K"}
        self.M = M
        self.S_max_ratio = S_max_ratio
        self.V_max_ratio = V_max_ratio
        self.grid_type = grid_type
        self.J_M_ratio = J_M_ratio
        self.K_M_ratio = K_M_ratio
        self.fixed_K = fixed_K

    def __repr__(self) :
        ret = "3dim rustic grid with M=" + str(self.M) + \
               ", S_max_ratio=" + str(self.S_max_ratio) + \
               ", V_max_ratio=" + str(self.V_max_ratio) + \
               ", grid_type=" + str(self.grid_type)
        if self.grid_type == "linear" :
            ret += ", J_M_ratio=" + str(self.J_M_ratio)
            ret += ", K_M_ratio=" + str(self.K_M_ratio)
        elif self.grid_type == "fixed_K" :
            ret += ", fixed_K=" + str(self.fixed_K)
        return ret

    def __eq__(self, other) :
        assert isinstance(other, Rustic_grid_3dim)
        return self.M == other.M \
               and self.S_max_ratio == other.S_max_ratio \
               and self.V_max_ratio == other.V_max_ratio \
               and self.grid_type == other.grid_type \
               and self.J_M_ratio == other.J_M_ratio \
               and self.K_M_ratio == other.K_M_ratio \
               and self.fixed_K == other.fixed_K

    def __hash__(self) :
        return hash((self.M, self.S_max_ratio, self.V_max_ratio, self.grid_type))

    def get_J(self) :
        if self.grid_type == "linear" or self.grid_type == "fixed_K" :
            J = self.J_M_ratio * self.M
        else :
            raise ValueError("grid type is invalid")
        return J

    def get_K(self) :
        if self.grid_type == "linear" :
            assert float(self.K_M_ratio * self.M).is_integer(), "The number of variance steps is not an integer."
            K = int(self.K_M_ratio * self.M)
        elif self.grid_type == "fixed_K" :
            assert self.fixed_K is not None, "fixed_K is not set"
            K = self.fixed_K
        else :
            raise ValueError("grid type is invalid")
        return K

    def cost(self):
        # returns int to avoid cost overflow
        return int(self.M) * int(self.get_J()) * int(self.get_K())


class Grid_2dim :
    """
    Class that contains the 2-dimensional (uniform) grid.
    Attributes:
    S_max       -- constructed as rustic_grid.S_max_ratio * S_0
    S           -- constructed from 0 to rustic_grid.S_max_ratio * S_0 with J steps
    dS          -- size of spacing
    dt          -- size of timestep
    rustic_grid -- rustic grid
    """
    def __init__(self, rustic_grid: Rustic_grid_2dim, S_0: float, T: float) :
        assert isinstance(rustic_grid, Rustic_grid_2dim)
        self.S_max = S_0 * rustic_grid.S_max_ratio
        self.dS = self.S_max / rustic_grid.get_J()
        self.dt = T / rustic_grid.M
        self.S = np.linspace(0, self.S_max, rustic_grid.get_J() + 1)
        self.rustic_grid = rustic_grid

    def get_lambda(self):
        return self.dt / self.dS


class Grid_3dim :
    """
    Class that contains the (uniform) 3-dimensional grid.
    Attributes:
    S_max       -- constructed as rustic_grid.S_max_ratio * S_0
    V_max       -- constructed as rustic_grid.V_max_ratio * theta (not V_0!)
    S           -- constructed from 0 to rustic_grid.S_max_ratio * S_0 with J steps
    V           -- constructed from 0 to rustic_grid.V_max_ratio * theta (not V_0!) with K steps
    dS, dt, dV  -- size of spacing, timestep and volatility step
    x1          -- ?
    x2          -- ?
    # Todo: implement get_lambda equivalent
    """
    def __init__(self, rustic_grid: Rustic_grid_3dim, S_0: float, V_0: float, T: float) :
        assert isinstance(rustic_grid, Rustic_grid_3dim)
        self.rustic_grid = rustic_grid
        self.S_max = S_0 * rustic_grid.S_max_ratio
        self.V_max = V_0 * rustic_grid.V_max_ratio
        self.dS = self.S_max / rustic_grid.get_J()
        self.dV = self.V_max / rustic_grid.get_K()
        self.dt = T / rustic_grid.M
        self.S = np.linspace(0, self.S_max, rustic_grid.get_J() + 1)
        self.V = np.linspace(0, self.V_max, rustic_grid.get_K() + 1)
        self.x1 = (np.ones((self.rustic_grid.get_K() + 1, 1)) * self.S).T
        self.x2 = np.ones((self.rustic_grid.get_J() + 1, 1)) * self.V
