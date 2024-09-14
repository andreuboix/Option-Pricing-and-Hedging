# filename: vanilla.py
# description: Vanilla securities objects
# author: Andreu Boix Torres, Giorgio Campisano
# date: 13-09-2024

"""
    Detailed description:
    This module defines all the securities objects for vanilla contracts that 
    use the pricing formulas of the library.
"""


# external modules
import math as m
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Callable



__all__ = [
    "Call",
    "Put"
]

class Vanilla:
    def __init__(self, S0: float, K: float, T: float, r: float) -> None:
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r

    def payoff(self):
        pass

    def price(self, method: Callable, *args, **kwargs) -> float:
        price_sim = method(*args, **kwargs)
        final_payoffs = self.payoff(price_sim[:, -1])
        return np.mean(final_payoffs*np.exp(self.r * self.T))

class Call(Vanilla):
    def __init__(self, S0: float, K: float, T: float, r: float) -> None:
        super().__init__(S0, K, T, r)

    def __repr__(self) -> str:
        return f"Call(S0={self.S0}, K={self.K}, T={self.T}, r={self.r})"
    
    def payoff(self, S: np.ndarray[float]) -> float:
        return np.maximum(S - self.K, 0)

class Put(Vanilla):
    def __init__(self, S0: float, K: float, T: float, r: float) -> None:
        super().__init__(S0, K, T, r)

    def __repr__(self) -> str:
        return f"Put(S0={self.S0}, K={self.K}, T={self.T}, r={self.r})"
 
    def payoff(self, S: np.ndarray[float]) -> float:
        return np.maximum(self.K - S, 0)
