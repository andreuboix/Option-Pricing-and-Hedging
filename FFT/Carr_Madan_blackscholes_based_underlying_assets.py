import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft


def CharExpBS(v,sig):    
    V = -sig**2/2*v**2 # := V(v)
    drift_rn = -(sig**2/2) # -V(-1j)
    V = drift_rn*1j*v + V 
    return V

def FFT_BS(Strike):
    # European Call - BS model
    S = 100
    T = 1
    r = 0.0367
    sig = 0.17801
    Npow = 20
    N = 2**Npow
    A = 1200
    eta = A/N
    v = np.arange(0, A, eta)
    v[0] = 1e-22
    Lambda = 2*np.pi/(N*eta)
    k = -Lambda*N/2 + Lambda*np.arange(N)
    
    
    # Fourier transform of z_k
    print("riskneutral check", np.exp(T*CharExpBS(-1j,sig)))

    Z_k = np.exp(1j*r*v*T)*(np.exp(T*CharExpBS(v-1j,sig))-1)/(1j*v*(1j*v+1))
    
    # Option price
    # Trapezoidal rule
    w = np.ones(N)
    w[0] = 0.5
    w[N-1] = 0.5
    x = w*eta*Z_k*np.exp(1j*np.pi*np.arange(N))
    
    z_k = np.real(fft(x)/np.pi)

    C = S*(z_k+np.maximum(1-np.exp(k-r*T),0))
    
    K = S*np.exp(k)
    
    # delete too small and too big strikes
    index = np.where((K > 0.1 * S) & (K < 3 * S))[0]
    C = C[index]
    K = K[index]

    # Plotting
    plt.plot(K, C) 
    plt.xlabel('strike')
    plt.ylabel('option price')
    plt.show()

    # Interpolation
    P_i = np.interp(Strike, K, C)
    return P_i


print(FFT_BS(100))