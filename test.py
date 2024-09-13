# filename: test.py
# description: testing environment for pricing frameworks
# author: Andreu Boix Torres, Giorgio Campisano
# date: 13-09-2024

"""
    Detailed description:
    This python module is a testing environment for all the pricing formulas 
    implemented in the library. It calls a nmber of different instruments from 
    the Securities module and applies the pricing formulas from the FFT and 
    MonteCarlo modules.
"""

# external modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# internal modules
from Securities.vanilla import Call, Put
from MonteCarlo.blackscholes_based_underlying_assets import *
from MonteCarlo.heston_based_underlying_assets import *
from FFT.Carr_Madan_blackscholes_based_underlying_assets import *


call = Call(S0=100, K=100, T=1, r=0.025)

bs_args = (
    1000, # N
    1000, # Nsim
    call.T, # T
    0.05, # mu
    0.1, # sigma
    call.S0, # S0
)

hes_args = (
    1000, # N
    1000, # Nsim
    call.T, # T
    0.01, # theta
    -.02, # k
    0.05, # epsilon
    call.r, # r
    call.S0, # X0
    0, # rho
    0.1 # v0
)

method1 = X_BS
method2 = Heston

c_bs = call.price(method1, *bs_args)
c_hes = call.price(method2, *hes_args)

print(f'--> {call}\n    Price - {method1.__name__}: {c_bs}')
print(f'--> {call}\n    Price - {method2.__name__}: {c_hes}')
