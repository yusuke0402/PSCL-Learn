from scipy.stats import gaussian_kde
from scipy.integrate import quad
import numpy as np
import pandas as pd

def estimate_lamda(sum_r,r_k,A,N_k_j):
    tmp=(A*r_k)/sum_r
    return min(tmp,N_k_j)