import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

S = 100
T = 1
r = 0.0367
mu = 1
v0, kappa, theta, sig, rho = [1e-02,  0.5e-02, 0.5e-01,  3.915e-02,  8e-02]
tau = 0.5
x = np.log(S)

def CharFunHeston(v):
    gamma = kappa-rho*sig*v*1j
    zeta = -1/2*(v**2+1j*v)
    psi = np.sqrt(gamma**2 - 2*(sig**2)*zeta)
    C = (-2*kappa*theta/sig**2)*(2*np.log( (2*psi-(psi-gamma)*(1-np.exp(-psi*tau))) / (2*psi)) + (psi-gamma)*tau)
    B = (2*zeta*(1-np.exp(-psi*tau))*v0)/(2*psi-(psi-gamma)*(1-np.exp(-psi*tau)))
    phi = np.exp(B+C)
    return phi


def FFT_Heston(Strike):
    # European Call - Heston model
    Npow = 20
    N = 2**Npow
    A = 1200
    eta = A/N
    v = np.arange(0, A, eta)
    v[0] = 1e-22
    Lambda = 2*np.pi/(N*eta)
    k = -Lambda*N/2 + Lambda*np.arange(N)
    
    # Fourier transform of z_k
    print("riskneutral check", CharFunHeston(-1j))

    Z_k = np.exp(1j*r*v*T)*(CharFunHeston(v-1j)-1)/(1j*v*(1j*v+1))
    
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


print(FFT_Heston(100))