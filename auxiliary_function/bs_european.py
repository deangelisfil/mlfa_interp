from numpy import pi, sqrt, log, exp
from scipy.stats import norm


def bs_call(S_0, sigma, r, T, K, opt) :
    """
    Black-Scholes European call option solution
    as defined in equation (3.17) on page 48 of
    The Mathematics of Financial Derivatives
    by Wilmott, Howison and Dewynne

    r     - interest rate
    sigma - volatility
    T     - time to maturity
    S_0     - asset value(s) at time 0
    K     - strike price(s)
    opt   - 'value', 'delta', 'gamma' or 'vega'
    V     - option value(s)
    """
    S_0 = S_0 + 1e-100  # avoids problems with S_0=0
    K = K + 1e-100  # avoids problems with K=0

    d1 = (log(S_0) - log(K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (log(S_0) - log(K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))

    if opt == 'value' :
        V = S_0 * norm.cdf(d1) - exp(-r * T) * K * norm.cdf(d2)
    elif opt == 'delta' :
        V = norm.cdf(d1)
    elif opt == 'gamma' :
        V = exp(-0.5 * d1 ** 2) / (sigma * sqrt(2 * pi * T) * S_0)
    elif opt == 'vega' :
        V = S_0 * (exp(-0.5 * d1 ** 2) / sqrt(2 * pi)) * (sqrt(T) - d1 / sigma) - \
            exp(-r * T) * K * (exp(-0.5 * d2 ** 2) / sqrt(2 * pi)) * \
            (-sqrt(T) - d2 / sigma)
    else :
        raise ValueError('invalid value for opt -- must be ''value'', ''delta'', ''gamma'', ''vega''')

    return V

def bs_put(S_0, sigma, r, T, K, opt):
    S_0 = S_0 + 1e-100  # avoids problems with S_0=0
    K = K + 1e-100  # avoids problems with K=0

    d1 = (log(S_0) - log(K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (log(S_0) - log(K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))

    if opt == 'value':
        V = exp(-r * T) * K * norm.cdf(-d2) - S_0 * norm.cdf(-d1)
    elif opt == 'delta':
        V = - bs_digital_put(S_0, sigma, r, T, K, "value")
    elif opt == 'gamma' :
        V = bs_dirac(S_0, sigma, r, T, K)
    else:
        raise NotImplementedError('invalid value for opt -- must be "value", "delta" or "gamma"')
    return V


def bs_digital_call(S_0, sigma, r, T, K, opt):
    """
    function V = digital_call(r,sigma,T,S_0,opt)

    Black-Scholes digital call option solution
    as defined on page 82 of
    The Mathematics of Financial Derivatives
    by Wilmott, Howison and Dewynne

    S_0         - asset value(s)
    sigma       - volatility
    r           - interest rate
    T           - time interval
    K           - strike
    opt         - 'value', 'delta' or 'gamma'
    V           - option value(s)
    """
    S_0 = S_0 + 1e-100  # avoids problems with S_0=0
    K = K + 1e-100  # avoids problems with K=0

    d2 = (log(S_0) - log(K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))

    if opt == 'value' :
        V = exp(-r * T) * norm.cdf(d2)
    elif opt == 'delta' :
        V = exp(-r * T) * (exp(-0.5 * d2 ** 2) / sqrt(2 * pi)) / (sigma * sqrt(T) * S_0)
    elif opt == 'gamma' :
        V = exp(-r * T) * (exp(-0.5 * d2 ** 2) / sqrt(2 * pi)) \
            * (-d2 / (sigma * sqrt(T) * S_0) - 1 / S_0) / (sigma * sqrt(T) * S_0)
    elif opt == 'vega' :
        V = exp(-r * T) * (exp(-0.5 * d2 ** 2) / sqrt(2 * pi)) * (-d2 / sigma - sqrt(T))
    else :
        raise ValueError('invalid value for opt -- must be "value", "delta", "gamma", "vega"')
    return V


def bs_digital_put(S_0, sigma, r, T, K, opt) :
    S_0 = S_0 + 1e-100  # avoids problems with S=0
    K = K + 1e-100  # avoids problems with K=0

    d2 = (log(S_0) - log(K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))

    if opt == 'value' :
        V = exp(-r * T) * (1 - norm.cdf(d2))
    elif opt == 'delta' :
        V = - exp(-r * T) * (exp(-0.5 * d2 ** 2) / sqrt(2 * pi)) / (sigma * sqrt(T) * S_0)
    elif opt == 'gamma' :
        V = - exp(-r * T) * (exp(-0.5 * d2 ** 2) / sqrt(2 * pi)) \
            * (-d2 / (sigma * sqrt(T) * S_0) - 1 / S_0) / (sigma * sqrt(T) * S_0)
    elif opt == 'vega' :
        V = - exp(-r * T) * (exp(-0.5 * d2 ** 2) / sqrt(2 * pi)) * (-d2 / sigma - sqrt(T))
    else :
        raise ValueError('invalid value for opt -- must be "value", "delta", "gamma", "vega"')
    return V


def bs_dirac(S_0, sigma, r, T, K):
    """
    function V = bs_fundamental_solution(S_0, sigma, r, T, K)

    Black Scholes PDE fundamental solution
    as defined on p.41 of the Financial Derivatives lecture notes
    by David Proemmel.
    """
    d2 = (log(S_0) - log(K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    V = exp(-r * T) / (K * sigma * sqrt(2 * pi * T)) * exp(- 0.5 * d2 ** 2)
    return V
