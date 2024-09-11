import numpy as np
from scipy.stats import expon, poisson, norm

def Heston(N, Nsim, T, theta, k, epsilon, r, X0, rho, V0):
    """
    Parameters
    ----------
    theta : 
        long term mean reversion.
    k : 
        constant that multiplies mean reversion.
    epsilon : 
        vol-of-vol: volatility of the volatility.
    r : 
        interest rate.
    X0 : 
        initial underlying asset price
    rho : 
        correlation between volatility and underlying asset brownian motions.
    V0 : 
        initial volatility value.

    Returns
    -------
        S_t Stock price.
    """
    # This is a CIR: example of Heston: rho < 0, f(y) = sqrt(y)
    # dS(t) = mu S(t)dt + sigma(t)S(t)dW1_t
    # sigma(t) = f(y(t)) where
    # dy(t) = k (theta - y(t))dt + epsilon*y(t)*dW2_t
    # dW1_t*dW2_t = rho*dt
    
    # theta: long term mean reversion
    # epsilon: vol-of-vol
    # rho: correlation between volatility and underlying asset brownian motions
    V = np.zeros((Nsim, N))
    X = np.zeros((Nsim, N))
    X[:,0] = X0
    V[:,0] = V0
    Feller_condition = (epsilon**2 - 2*k*theta)
    dt = T / N
    if Feller_condition > 0: # QE Scheme
        Psi_cutoff = 1.5
        for i in range(N): # Discretise V (volatility) and calculate m, Psi, s2
            m = theta + (V[:,i] - theta)*np.exp(-k*dt)
            m2 = m**2
            s2 = V[:,i]*epsilon**2*np.exp(-k*dt)*(1-np.exp(-k*dt))/k + theta*epsilon**2*(1-np.exp(-k*dt))**2/(2*k)
            Psi = (s2)/(m2)
            index = np.where(Psi_cutoff < Psi)[0]
            
            # Exponential approx scheme if Psi > Psi_cutoff
            p_exp = (Psi[index]-1)/(Psi[index]+1)
            beta_exp = (1-p_exp)/m[index]
            U = np.random.rand(np.size(index))
            V[index,i+1] = (np.log((1-p_exp)/(1-U)/beta_exp)*(U>p_exp))
            
            # Quadratic approx scheme if 0 < Psi < Psi_cutoff
            index = np.where(Psi <= Psi_cutoff)[0]
            invPsi = 1/Psi[index]
            b2_quad = 2*invPsi - 1 + np.sqrt(2*invPsi)*np.sqrt(2*invPsi-1)
            a_quad  = m[index]/(1+b2_quad)
            V[index, i+1] = a_quad*(np.sqrt(b2_quad)+ np.random.randn(np.size(index)))**2
            
        # Central discretisation scheme
        gamma1 = 0.5
        gamma2 = 0.5
        k0 = r*dt -rho*k*theta*dt/epsilon
        k1 = gamma1*dt*(k*rho/epsilon-0.5)-rho/epsilon
        k2 = gamma2*dt*(k*rho/epsilon-0.5)+rho/epsilon
        k3 = gamma1*dt*(1-rho**2)
        k4 = gamma2*dt*(1-rho**2)
        for i in range(N):
            X[:,i+1] = np.exp(np.log(X[:,i]))+k0+k1*V[:,i]+k2*V[:,i+1]+np.sqrt(k3*V[:,i])+k4*V[:,i+1]*np.random.randn(1,Nsim)
        return X
    else:
        mu = [0, 0]
        VC = [[1,rho],[rho,1]]
        for i in range(N):
            Z = np.random.multivariate_normal(mu,VC,Nsim)
            X[:,i+1] = X[:,i] + (r-V[:,i]/2)*dt+np.sqrt(V[:,i]*dt)*Z[:,1]
            V[:,i+1] = V[:,i] + k*(theta-V[:,i])*dt+epsilon*np.sqrt(V[:,i]*dt)*Z[:,2]
        return np.exp(X)
def MD_Heston(N, Nsim, T, theta, k, epsilon, r, X0, rho, V0):
    """
    Parameters
    ----------
    N : int
        Number of time steps.
    Nsim : int
        Number of simulations.
    T : float
        Time horizon.
    theta : ndarray
        Long-term mean reversion for each dimension.
    k : ndarray
        Constant that multiplies mean reversion for each dimension.
    epsilon : ndarray
        Volatility of volatility for each dimension.
    r : float
        Risk-free interest rate.
    X0 : ndarray
        Initial underlying asset price for each dimension.
    rho : ndarray
        Correlation between volatility and underlying asset brownian motions for each dimension.
    V0 : ndarray
        Initial volatility value for each dimension.

    Returns
    -------
    S_t : ndarray
        Array of shape (Nsim, N, d) representing the simulated paths for each dimension.
    """
    
    # Check that input arrays have the correct shape
    d = len(X0)
    
    # Initialize arrays to store simulated paths
    V = np.zeros((Nsim, N, d))
    X = np.zeros((Nsim, N+1, d))
    X[:,:,0] = X0
    V[:,:,0] = V0
    
    # Feller condition
    Feller_condition = (epsilon**2 - 2*k*theta)
    dt = T / N
    
    if Feller_condition > 0: # QE Scheme
        Psi_cutoff = 1.5
        for i in range(N): # Discretise V (volatility) and calculate m, Psi, s2 for each dimension
            m = theta + (V[:,i,:] - theta)*np.exp(-k*dt)
            m2 = m**2
            s2 = V[:,i,:]*epsilon**2*np.exp(-k*dt)*(1-np.exp(-k*dt))/k + theta*epsilon**2*(1-np.exp(-k*dt))**2/(2*k)
            Psi = (s2)/(m2)
            index = np.where(Psi_cutoff < Psi)
            
            # Exponential approx scheme if Psi > Psi_cutoff
            p_exp = (Psi[index]-1)/(Psi[index]+1)
            beta_exp = (1-p_exp)/m[index]
            U = np.random.rand(np.size(index))
            V[index,i+1,:] = (np.log((1-p_exp)/(1-U)/beta_exp)*(U>p_exp))
            
            # Quadratic approx scheme if 0 < Psi < Psi_cutoff
            index = np.where(Psi <= Psi_cutoff)
            invPsi = 1/Psi[index]
            b2_quad = 2*invPsi - 1 + np.sqrt(2*invPsi)*np.sqrt(2*invPsi-1)
            a_quad  = m[index]/(1+b2_quad)
            V[index, i+1,:] = a_quad*(np.sqrt(b2_quad)+ np.random.randn(np.size(index), d))**2
        
        # Central discretisation scheme
            gamma1 = 0.5
            gamma2 = 0.5
            k0 = r*dt -rho*k*theta*dt/epsilon
            k1 = gamma1*dt*(k*rho/epsilon-0.5)-rho/epsilon
            k2 = gamma2*dt*(k*rho/epsilon-0.5)+rho/epsilon
            k3 = gamma1*dt*(1-rho**2)
            k4 = gamma2*dt*(1-rho**2)
            for i in range(N):
                X[:,i+1] = np.exp(np.log(X[:,i]))+k0+k1*V[:,i]+k2*V[:,i+1]+np.sqrt(k3*V[:,i])+k4*V[:,i+1]*np.random.randn(1,Nsim)
            return X
    else:
            mu = [0, 0]
            VC = [[1,rho],[rho,1]]
            for i in range(N):
                Z = np.random.multivariate_normal(mu,VC,Nsim)
                X[:,i+1] = X[:,i] + (r-V[:,i]/2)*dt+np.sqrt(V[:,i]*dt)*Z[:,1]
                V[:,i+1] = V[:,i] + k*(theta-V[:,i])*dt+epsilon*np.sqrt(V[:,i]*dt)*Z[:,2]
            return np.exp(X)
