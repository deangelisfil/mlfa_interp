from data_generation.data_generator.data_generator_abstract import Data_generator_abstract, cost
import numpy as np
from abc import ABC, abstractmethod
import matlab.engine


class Mlfa_interp_generator(ABC):
    """
    Class that generates data for mlfa_interp.
    """
    def __init__(self,
                 numerical_method_L: list,
                 data_generator: Data_generator_abstract,
                 Lstart,
                 kstart):
        """
        Inputs:
            numerical_method_L:     list of numerical methods that are used to generate the data
            data_generator:         instance of Data_generator_abstract from which the data is generated from
            how:                    determines how to discretize the interpolation grid. Either "tensor" or "sparse".
        """
        assert len(numerical_method_L) == len(set(numerical_method_L)), "Entries of numerical_method_L are not unique."
        assert kstart >= 1, "At least 3 points in each direction, so that we can test with 2 in each direction."
        self.numerical_method_L = numerical_method_L
        assert data_generator.how == "grid", "MLFA interp generator requires samples on the grid."
        self.data_generator = data_generator
        assert Lstart <= len(numerical_method_L), "Lstart > len(numerical_method_L) is invalid."
        self.Lstart = Lstart
        self.kstart = kstart
    @abstractmethod
    def get_kl(self):
        pass

    @abstractmethod
    def reset(self, eng: matlab.engine):
        pass

    @abstractmethod
    def __call__(self, l: int, eng: matlab.engine):
        pass

class Mlfa_tensor_interp_generator(Mlfa_interp_generator):
    def __init__(self,
                 numerical_method_L: list,
                 data_generator: Data_generator_abstract,
                 Lstart: int = 0,
                 kstart: int = 1):
        super().__init__(numerical_method_L, data_generator, Lstart, kstart)
        self.state = {l: tuple() for l in range(len(self.numerical_method_L) - self.Lstart)}
        self.reset()

    def get_N(self, k: int):
        """
        Returns the number of interpolation points on tensor grid of level k.
        """
        N_1_dim = 2 ** k + 1
        dim = self.data_generator.get_dim(drop_constant_parameter=True)
        return N_1_dim ** dim

    def get_kl(self):
        return np.array([state[0] for state in self.state.values()])

    def reset(self, eng: matlab.engine = None):
        """
        Resets self.state to self.kstart values.
        to the interpolation grid with two points in each direction.
        **kwargs are just ignored.
        """
        assert eng is None, "eng is not used."
        L_effective = len(self.numerical_method_L) - self.Lstart
        N = self.get_N(k=self.kstart-1)
        grid = self.data_generator.get_grid(N=N, drop_constant_parameter=True)
        for l in range(L_effective):
            _, y, err, _ = \
                self.data_generator.err_numerical_approximation(l=l,
                                                                N=N,
                                                                numerical_method_L=self.numerical_method_L[self.Lstart:])
            self.state[l] = self.kstart, grid, y, err

    def __call__(self, l: int, eng: matlab.engine = None):
        """
        Returns interpolation grid on level l and then increases the accuracy level of the interpolation grid by 1.
        Also stores the previous grid, y and err.
        **kwargs is not used. This is here to make the interface compatible with Mlfa_sparse_interp_generator.
        """
        assert eng is None, "eng is not used."
        assert self.Lstart + l <= len(self.numerical_method_L), "l > Lmax is invalid."
        k, grid_train, y_train, err_train = self.state[l]
        N = self.get_N(k=k)

        #
        # generate validation grid
        #

        x, y, err, cst = \
            self.data_generator.err_numerical_approximation(l=l,
                                                            N=N,
                                                            numerical_method_L=self.numerical_method_L[self.Lstart:])

        #
        # Update the state
        #

        grid = self.data_generator.get_grid(N=N, drop_constant_parameter=True)
        self.state[l] = k + 1, grid, y, err

        return grid_train, y_train, err_train, x, y, err, cst

class Mlfa_sparse_interp_generator(Mlfa_interp_generator):
    def __init__(self,
                 numerical_method_L: list,
                 data_generator: Data_generator_abstract,
                 Lstart: int = 0,
                 kstart: int = 1):
        super().__init__(numerical_method_L, data_generator, Lstart, kstart)
        self.kl = np.ones(len(self.numerical_method_L) - self.Lstart, dtype=float) * self.kstart

    def get_kl(self):
        return self.kl

    def set_Lstart(self, Lstart: int, eng: matlab.engine):
        self.Lstart = Lstart
        eng.workspace['mlfa_interp_generator.Lstart'] = self.Lstart

    def reset(self, eng: matlab.engine):
        """
        Input engine:
        - numerical_method_L
        - Lstart
        """
        eng.eval("z = cell(1, py.len(mlfa_interp_generator.numerical_method_L) - mlfa_interp_generator.Lstart);", nargout=0)
        self.kl = np.ones(len(self.numerical_method_L) - self.Lstart, dtype=float) * self.kstart


    def __call__(self, l: int, eng: matlab.engine):
        """
        Input engine:
        - f:             function handle
        - dim:           dimension of the parameter space
        - range:         range of the parameter space
        - options:       options for the sparse interpolation.
                         Tyically, options = spset('GridType', 'Maximum', 'Vectorized', 'on')
        - z:             L+1 dimensional cell with the current values of the sparse grid interpolations.
        - target:        "delta_f" or "f"

        return N_train, N, max_err, cst

        Output engine updates:
        - options:       now includes the new values for z{l+1} as previous result, and k as MinDepth and MaxDepth
        - z:             new z{l+1} for the interpolation grid at level kl[l]
        - x:         matrix Nxdim, where N is the number of interpolation grid points at level k
        - err:         vector Nx1 with the function values at the interpolation grid points
        - z_train:       if kl[l] > 1: the previous z{l+1}; stored to estimate the interpolation error with spinterp
                         if kl[l] == 1: (1x2) cell with grid_train and err_train; stored to estimate the interpolation error with tensor_grid_interp
        """

        #
        # construct z_train
        #

        eng.workspace['l'] = l
        if eng.eval("options.GridType;", nargout=1) == "Maximum":
            eng.workspace['k'] = self.kl[l] - 1  # spinterp: k==0 3 points in each direction
                                                 # mlfa: k==1 3 points in each direction
        elif eng.eval("options.GridType;", nargout=1) == "Clenshaw-Curtis":
            eng.workspace['k'] = self.kl[l] + 1  # spinterp: k==2 more than 2 points in each direction
                                                 # mlfa: k==1 3 points in each direction
        else:
            ValueError("options.GridType is neither Maximum nor Clenshaw-Curtis grid.")

        if self.kl[l] == 1 and eng.eval("options.GridType;", nargout=1) == "Maximum":
            # construct -1 interpolation grid in python
            N_train = 2 ** self.data_generator.get_dim(drop_constant_parameter=True) # 2 points in each direction,
            grid_train = self.data_generator.get_grid(N=N_train, drop_constant_parameter=True)
            x_train, y_train, err_train, _ = \
                self.data_generator.err_numerical_approximation(l=l,
                                                                N=N_train,
                                                                numerical_method_L=self.numerical_method_L[self.Lstart:])
            eng.workspace['grid_m1'] = grid_train
            if eng.eval("target", nargout=1) == "delta_f":
                eng.workspace['err_m1'] = err_train
            elif eng.eval("target", nargout=1) == "f":
                eng.workspace['err_m1'] = y_train
            else:
                raise ValueError("target not recognized.")
        elif self.kl[l] == 1 and eng.eval("options.GridType;", nargout=1) == "Clenshaw-Curtis":
            eng.eval("options = spset(options, 'PrevResults', z{l+1}, 'MinDepth', k-1, 'MaxDepth', k-1);", nargout=0)
            eng.eval("z{l+1} = spvals(f, dim, range, options, l);", nargout=0)
            eng.eval("z_train = z{l+1};", nargout=0)
            N_train = eng.eval("z_train.nPoints", nargout=1)
        else:
            eng.eval("z_train = z{l+1};", nargout=0)
            N_train = eng.eval("z_train.nPoints", nargout=1)

        #
        # run spvals and construct output x_val, y_val
        #

        eng.eval("options = spset(options, 'PrevResults', z{l+1}, 'MinDepth', k, 'MaxDepth', k);", nargout=0)
        eng.eval("z{l+1} = spvals(f, dim, range, options, l);", nargout=0)
        if eng.eval("options.GridType;", nargout=1) == "Maximum":
            eng.eval("x = vertcat(z{l+1}.grid{:});", nargout=0)
        else:
            eng.eval("x = z{l+1}.grid;", nargout=0)
        eng.eval("err = vertcat(z{l+1}.fvals{:});", nargout=0)
        max_err = eng.eval("max(abs(err));", nargout=1)
        N = eng.eval("z{l+1}.nPoints", nargout=1)
        cst = cost(l=l, N=N, numerical_method_L=self.numerical_method_L[self.Lstart:])

        #
        # Update kl
        #

        self.kl[l] += 1

        return N_train, N, max_err, cst