import numpy as np
from scipy.integrate import quad

def heston_call(S_0: float,
                V_0: float,
                r: float,
                kappa: float,
                theta: float,
                omega: float,
                rho: float,
                T: float,
                K: float):
    """
    The function
        V = heston_call(r, kappa, theta, omega, rho, S, K, V0, T)
    gives the analytic solution for European call option in Heston model,
    using the algorithm in "Not-so-complex logarithms in the Heston model"
    by Christian Kahl and Peter Jaekel
        http: // www.math.uni - wuppertal.de / ~kahl / publications / NotSoComplexLogarithmsInTheHestonModel.pdf
        http: // www.btinternet.com / ~pjaeckel / NotSoComplexLogarithmsInTheHestonModel.pdf
    The code follows Mike Giles' Matlab code in heston_call.m
    Input:
    r - interest rate
    kappa - mean reversion rate
    theta - mean reversion volatility
    omega - vol - of - vol coefficient
    rho - correlation factor
    S_0 - initial asset value(s)
    K - strike price
    V_0 - initial volatility or variance
    T - time to maturity
    Output:
    V - option value(s)
    Notes:
        Achieves better accuracy 1e-7, otherwise throws an error.
    """

    V, err, infodict, message = quad(lambda x: integrand(x, S_0, V_0, r, kappa, theta, omega, rho, T, K),
                                     0,
                                     1,
                                     epsabs=1e-16,
                                     epsrel=0,  # stopping criterion does not depend on relative error
                                     limit=1000,
                                     full_output=1)

    assert err < 1e-7, "absolute error of the numerical integration is {0} and is too big.\n" \
                        "Integration message: {1}\n" \
                        "The parameters are: S_0={2}, V_0={3}, r={4}, kappa={5}, theta={6}, " \
                        "omega={7}, rho={8}, T={9}, K={10}".format(err,
                                                                  message,
                                                                  np.round(S_0, 2),
                                                                  np.round(V_0, 2),
                                                                  np.round(r, 2),
                                                                  np.round(kappa, 2),
                                                                  np.round(theta, 2),
                                                                  np.round(omega, 2),
                                                                  np.round(rho, 2),
                                                                  np.round(T, 2),
                                                                  np.round(K, 2))
    V = np.exp(-r*T) * V
    return V


def integrand(x, S_0, V_0, r, kappa, theta, omega, rho, T, K):
    """
    input:
    x - np.array
    """
    x = np.maximum(1e-20, np.minimum(x, 1 - 1e-10))
    Cinf = np.sqrt(1 - rho**2) * (V_0 + kappa * theta * T) / omega
    u = - np.log(x) / Cinf
    F = S_0 * np.exp(r * T)
    f = []

    for ipass in range(2):
        um = u - 1j * (1 - ipass)
        d = np.sqrt((rho * omega * um * 1j - kappa)**2 + omega**2 * (um * 1j + um**2))
        c = (kappa - rho * omega * um * 1j + d) / (kappa - rho * omega * um * 1j - d)
        tc = np.angle(c)
        GD = c - 1
        m = np.floor((tc + np.pi) / (2 * np.pi))
        GN = c * np.exp(1j * np.imag(d) * T) - np.exp(-np.real(d) * T)
        n = np.floor((tc + np.imag(d) * T + np.pi) / (2 * np.pi))
        lnG = np.real(d) * T + np.log(abs(GN) / abs(GD)) + 1j * (np.angle(GN) - np.angle(GD) + 2 * np.pi * (n - m))
        D = ((kappa - rho * omega * um * 1j + d) / omega**2) * ((np.exp(d * T) - 1) / (c * np.exp(d * T) - 1))
        C = ((kappa * theta) / omega**2) * ((kappa - rho * omega * um * 1j + d) * T - 2 * lnG)
        phi = np.exp(C + D * V_0 + 1j * um * np.log(F))
        f.append(np.real(np.exp(-1j * u * np.log(K)) * phi / (1j * u)))

    return 0.5 * (F - K) + (f[0] - K * f[1]) / (x * np.pi * Cinf)
