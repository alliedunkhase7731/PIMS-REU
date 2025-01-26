import numpy as np

def eV2kms(eV, Z, A):
    return np.sqrt(eV * (1 / A) * (170.**2) / 150.)

def ZV2kms(V, Z, A):
    return np.sqrt(V * (Z / A) * (170.**2) / 150.)