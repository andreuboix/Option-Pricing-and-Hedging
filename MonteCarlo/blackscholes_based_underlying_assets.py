import numpy as np
from scipy.stats import expon, poisson, norm

__all__ = [
    "X_BS",
    "MD_X_BS",
    "BM_inc",
    "MD_BM_inc",
    "X_BS_Kou",
    "MD_X_BS_Kou",
    "X_BS_Merton",
    "MD_X_BS_Merton"
]

def X_BS(N: int, Nsim: int, T: float, mu: float, sigma: float, S0: float) -> np.ndarray:
    """
    Parameters
    ----------
    N : int
        Number of time steps.
    Nsim : int
        Number of simulations.
    T : float
        Time horizon.
    mu : float
        BS parameter, drift.
    sigma : float
        BS parameter, volatility.
    S0 : float
        initial Stock price.
    """
    XT = mu * T + sigma * np.sqrt(T) * np.random.randn(Nsim, 1)
    ST = S0 * np.exp(XT)
    dt = T / N
    t = np.linspace(0, T, N+1)
    X = np.zeros((N+1,Nsim))
    for i in range(1, N+1):
        X[i,:] = X[i-1,:] + mu*dt + sigma*np.sqrt(dt)*np.random.randn(Nsim)
    return S0*np.exp(X)


def MD_X_BS(N: int, Nsim: int, T: float, mu: float, sigma: float, S0: float) -> np.ndarray:
    """
    Parameters
    ----------
    N : int
        Number of time steps.
    Nsim : int
        Number of simulations.
    T : float
        Time horizon.
    mu : float
        BS parameter, drift vector of length d.
    sigma : float
        BS parameter, volatility matrix of length d.
    S0 : float
        initial Stock price vector of length d.
    """
    d = len(S0)
    XT = mu * T + np.dot(np.linalg.cholesky(sigma), np.random.normal(size=(d, Nsim))).T * np.sqrt(T)
    ST = S0 * np.exp(XT)
    dt = T / N
    t = np.linspace(0, T, N+1)
    X = np.zeros((N+1, Nsim, d))
    X[0,:,:] = np.log(S0)
    for i in range(1, N+1):
        dW = np.dot(np.linalg.cholesky(sigma), np.random.normal(size=(d, Nsim))) * np.sqrt(dt)
        X[i,:,:] = X[i-1,:,:] + (mu - 0.5 * np.diag(sigma)) * dt + dW.T
    
    return np.exp(X)


def BM_inc(N, K_U, J, T):
    dt = T/N
    W = np.zeros((N+1,K_U,J))
    for i in range(1, N+1):
        W[i,:,:] = W[i-1,:,:] + np.sqrt(dt)*np.random.randn(K_U, J)
    return W

def MD_BM_inc(N,K_U,J,T,d):
    dt = T/N
    W = np.zeros((N+1,K_U,J,d))
    for i in range(1, N+1):
        W[i,:,:,:] = W[i-1,:,:,:] + np.sqrt(dt)*np.random.randn(K_U, J,d)
    return W

    
def X_BS_Kou(N,Nsim, T,mu,sigma, S0, lambdat, p, lambdap, lambdam):
    """
    Parameters
    ----------
    mu : 
        BS parameter, drift.
    sigma : 
        BS parameter, volatility.
    S0 : 
        initial Stock price.
    lambdat : 
        The distribution of the counting process for the jump distribution is
        N(t) ~ Pois(lambdat * t)
    p, lambdap, lambdam : 
        Kou parameter. The jump distribution for Kou is Exp(1/lambdap)*(p) + Exp(1/lambdam)*(1-p).
        The probability of having a positive jump is p and of having a negative jump is 1-p

    Returns
    -------
        Stock price
    """
    dt = T / N
    t = np.linspace(0, T, N+1)
    X = np.zeros((N+1,Nsim))
    NT = poisson.ppf(np.random.rand(Nsim, 1), lambdat * T)
    for j in range(Nsim):
        Tj = np.sort(T * np.random.rand(int(NT[j])))
        Z = np.random.randn(1, N)
        for i in range(1, N+1):
            X[i,j] = X[i-1,j] + mu * dt + sigma * np.sqrt(dt) * Z[0, i-1]
            for k in range(int(NT[j])):
                if (int(Tj[k]) > int(t[i-1])) & (int(Tj[k]) <= int(t[i])):
                    uniform_value = np.random.rand(1)
                    if uniform_value < p:
                        Y = expon.ppf(uniform_value, 1/lambdap)
                    else:
                        Y = -expon.ppf(uniform_value, 1/lambdam)
                    X[i,j] = X[i,j] + Y
    
    return S0*np.exp(X)

def MD_X_BS_Kou(N, Nsim, T, mu, sigma, S0, lambdat, p, lambdap, lambdam, d):
    """
    Parameters
    ----------
    mu : 
        BS parameter, drift.
    sigma : 
        BS parameter, volatility.
    S0 : 
        initial Stock price.
    lambdat : 
        The distribution of the counting process for the jump distribution is
        N(t) ~ Pois(lambdat * t)
    p, lambdap, lambdam : 
        Kou parameter. The jump distribution for Kou is Exp(1/lambdap)*(p) + Exp(1/lambdam)*(1-p).
        The probability of having a positive jump is p and of having a negative jump is 1-p
    d : 
        dimension of the Black-Scholes process

    Returns
    -------
        Stock price
    """
    dt = T / N
    t = np.linspace(0, T, N+1)
    X = np.zeros((N+1,Nsim, d))
    NT = poisson.ppf(np.random.rand(Nsim, 1), lambdat * T)
    for j in range(Nsim):
        Tj = np.sort(T * np.random.rand(int(NT[j])))
        Z = np.random.randn(N, d)
        for i in range(1, N+1):
            X[i,j, :] = X[i-1,j,:] + mu * dt + sigma * np.sqrt(dt) * Z[i-1, :]
            for k in range(int(NT[j])):
                if (int(Tj[k]) > int(t[i-1])) & (int(Tj[k]) <= int(t[i])):
                    uniform_value = np.random.rand(1)
                    if uniform_value < p:
                        Y = expon.ppf(uniform_value, 1/lambdap)
                    else:
                        Y = -expon.ppf(uniform_value, 1/lambdam)
                    J = np.zeros(d)
                    for l in range(d):
                        J[l] = Y * np.random.randn(1)
                    X[i,j,:] = X[i,j,:] + J
    
    return S0*np.exp(X)

def X_BS_Merton(N,Nsim, T,mu,sigma, S0, lambdat, muJ, deltaJ):
    """
    Parameters
    ----------
    mu : 
        BS parameter, drift.
    sigma : 
        BS parameter, volatility.
    S0 : 
        initial Stock price.
    lambdat : 
        The distribution of the counting process for the jump distribution is
        N(t) ~ Pois(lambdat * t)
    muJ : 
        Merton parameter. The jump distribution for Merton is Normal(muJ, deltaJ^2).
    deltaJ : 
        Merton parameter. The jump distribution for Merton is Normal(muJ, deltaJ^2).

    Returns
    -------
        Stock price
    """
    dt = T / N
    t = np.linspace(0, T, N+1)
    X = np.zeros((N+1,Nsim))
    NT = poisson.ppf(np.random.rand(Nsim, 1), lambdat * T)
    for j in range(Nsim):
        Tj = np.sort(T * np.random.rand(int(NT[j])))
        Z = np.random.randn(1, N)
        for i in range(1, N+1):
            X[i,j] = X[i-1,j] + mu * dt + sigma * np.sqrt(dt) * Z[0, i-1]
            for k in range(int(NT[j])):
                if (int(Tj[k]) > int(t[i-1])) & (int(Tj[k]) <= int(t[i])):
                    Y = norm.ppf(q = np.random.rand(1), loc= muJ, scale= deltaJ)
                    X[i,j] = X[i,j] + Y
    return S0*np.exp(X)

def MD_X_BS_Merton(N, Nsim, T, mu, sigma, S0, lambdat, muJ, deltaJ):
    """
    Parameters
    ----------
    mu : 
        BS parameter, drift.
    sigma : 
        BS parameter, volatility.
    S0 : 
        initial Stock price.
    lambdat : 
        The distribution of the counting process for the jump distribution is
        N(t) ~ Pois(lambdat * t)
    muJ : 
        Merton parameter. The jump distribution for Merton is Normal(muJ, deltaJ^2).
    deltaJ : 
        Merton parameter. The jump distribution for Merton is Normal(muJ, deltaJ^2).

    Returns
    -------
        Stock price
    """
    dt = T / N
    t = np.linspace(0, T, N+1)
    d = len(mu)
    X = np.zeros((N+1,Nsim, d))
    NT = poisson.ppf(np.random.rand(Nsim, 1), lambdat * T)
    for j in range(Nsim):
        Tj = np.sort(T * np.random.rand(int(NT[j])))
        Z = np.random.randn(d, N)
        for i in range(1, N+1):
            X[i,j,:] = X[i-1,j,:] + mu * dt + sigma * np.sqrt(dt) * Z[:, i-1]
            for k in range(int(NT[j])):
                if (int(Tj[k]) > int(t[i-1])) & (int(Tj[k]) <= int(t[i])):
                    Y = np.random.normal(muJ, deltaJ, d)
                    X[i,j,:] = X[i,j,:] + Y
                    
    X[0,:,:] = np.zeros((Nsim,d))
    return S0*np.exp(X)
