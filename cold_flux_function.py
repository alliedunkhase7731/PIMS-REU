import numpy as np
import math
from conversions import eV2kms, ZV2kms

def cold_flux_function(Vlo, Vhi, TeV, Z, A, uz):
    wkms = eV2kms(TeV, Z, A)
    vlokms = ZV2kms(Vlo, Z, A)
    vhikms = ZV2kms(Vhi, Z, A)
    ulim = (1. / wkms) * (vhikms - uz)
    llim = (1. / wkms) * (vlokms - uz)
    if uz == 0:
        return 0 
    else:
        Qv = wkms / (uz * np.sqrt(np.pi))
        return 0.5 * (math.erf(ulim) - math.erf(llim) - Qv * (np.exp(-ulim ** 2) - np.exp(-llim ** 2)))
