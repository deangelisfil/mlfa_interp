This package implements the numerical results for the work on "Multilevel Function Approximation" of Filippo De Angelis, Prof. Mike Giles, and Prof. Christoph Reisinger. 

The work offers new ideas on the approximation of functions $\xi \mapsto f(\xi)$, where each function evaluation corresponds to either a functional of the solution of a PDE, with parametric dependence on $\xi$, or the expected value of a functional of the solution of an SDE, again with a parametric dependence on $\xi$. In both cases, exact sampling of $f(\xi)$ is not possible, and greater accuracy comes at a higher computational cost.

The key idea to improve the computational cost for a given accuracy is a multilevel representation of the function $f$. Coarse levels use inaccurate approximations of $f(\xi)$ at a large number of points, whereas fine levels use very accurate approximations at a very limited number of points.

As an application of the multilevel approach, we consider learning the function that maps the parameters of the model and of the financial product to the price of the financial product. 

In the simple case of approximating the Black-Scholes call price function $(\sigma, r, C) \mapsto C(\sigma, r, K)$ using piecewise linear interpolation on tensor grids and second-order accurate finite difference methods, the computational cost to achieve accuracy $\bigO(\epsilon)$ is reduced from $\bigO(\epsilon^{-5/2})$ to $\bigO(\epsilon^{-3/2})$.

For one-layer neural networks and second-order accurate finite difference methods, the computational cost to achieve accuracy $\mathcal{O}(\varepsilon)$ is reduced from $\mathcal{O}(\varepsilon^{-4 - d_X / 2})$ to  $\mathcal{O}(\varepsilon^{-4})$, where $d_X$ is the dimension of the underlying pricing PDE.
