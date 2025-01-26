import numpy as np
from constants import e0_pC, cm2km
from cold_flux_function import cold_flux_function
from pims_r_theta import pims_r_theta

def calculate_currents(nwin, Vlow, Vhigh, theta, uz, TeV, necm3, Xp, OSratio, ZO2, ZS2, ZS3, phi, noise_level):
    eA = e0_pC * cm2km * np.pi
    
    # Electron density (input)
    ne = necm3  # Electron density (in electrons per cm^3)
    
    # Number densities of ions from the electron density and composition
    nH = Xp * ne
    nO = ne * (1. - Xp) / (1. + (1. / OSratio))
    nS = nO / OSratio
    
    # Fractional abundances of different ionization states
    nO2 = nO * ZO2
    nO1 = nO - nO2
    nS2 = nS * ZS2
    nS3 = nS * ZS3
    nS1 = nS - (nS2 + nS3)

    # Atomic masses (H = 1 amu, O = 16 amu, S = 32 amu)
    # Units in amu / cm^3
    m_H, m_O, m_S = 1, 16, 32
    
    # Mass densities (rho = n * m)
    rho_H = nH * m_H
    rho_O1 = nO1 * m_O
    rho_O2 = nO2 * m_O
    rho_S1 = nS1 * m_S
    rho_S2 = nS2 * m_S
    rho_S3 = nS3 * m_S
    rho_total = rho_H + rho_O1 + rho_O2 + rho_S1 + rho_S2 + rho_S3
    
    # Voltage windows and currents initialization
    Vwindows = np.logspace(np.log10(Vlow), np.log10(Vhigh), num=nwin)
    IH, IO1, IO2, IS1, IS2, IS3 = np.zeros(nwin), np.zeros(nwin), np.zeros(nwin), np.zeros(nwin), np.zeros(nwin), np.zeros(nwin)
    
    for window in range(nwin - 1):
        IH[window] = eA * nH * uz * np.real(cold_flux_function(Vwindows[window], Vwindows[window + 1], TeV, 1., 1., uz))
        IO1[window] = eA * nO1 * uz * np.real(cold_flux_function(Vwindows[window], Vwindows[window + 1], TeV, 1., 16., uz))
        IO2[window] = 2 * eA * nO2 * uz * np.real(cold_flux_function(Vwindows[window], Vwindows[window + 1], TeV, 2., 16., uz))
        IS1[window] = eA * nS1 * uz * np.real(cold_flux_function(Vwindows[window], Vwindows[window + 1], TeV, 1., 32., uz))
        IS2[window] = 2 * eA * nS2 * uz * np.real(cold_flux_function(Vwindows[window], Vwindows[window + 1], TeV, 2., 32., uz))
        IS3[window] = 3 * eA * nS3 * uz * np.real(cold_flux_function(Vwindows[window], Vwindows[window + 1], TeV, 3., 32., uz))
    
    # Total current without noise
    Itot = IH + IO1 + IO2 + IS1 + IS2 + IS3
    
    # Add noise to the total current
    Itot_noise = np.sqrt(Itot**2 + (noise_level * np.random.normal(size=Itot.shape))**2)

    
    return Vwindows, Itot_noise, rho_total