import numpy as np
from constants import e0_pC, cm2km
from cold_flux_function import cold_flux_function
from pims_r_theta import pims_r_theta

def calculate_currents(nwin, Vlow, Vhigh, theta, uz, TeV, necm3, Xp, OSratio, ZO2, ZS2, ZS3, phi):
    eA = e0_pC * cm2km * np.pi
    nH = Xp * necm3
    nO = necm3 * (1. - Xp) / (1. + (1. / OSratio))
    nS = nO / OSratio
    nO2 = nO * ZO2
    nO1 = nO - nO2
    nS2 = nS * ZS2
    nS3 = nS * ZS3
    nS1 = nS - (nS2 + nS3)

    Vwindows = np.logspace(np.log10(Vlow), np.log10(Vhigh), num=nwin)
    IH = np.zeros(nwin)
    IO1 = np.zeros(nwin)
    IO2 = np.zeros(nwin)
    IS1 = np.zeros(nwin)
    IS2 = np.zeros(nwin)
    IS3 = np.zeros(nwin)
    
    for window in range(nwin - 1):
        IH[window] = 1. * eA * nH * uz * np.real(cold_flux_function(Vwindows[window], Vwindows[window + 1], TeV, 1., 1., uz)) * pims_r_theta(theta)
        IO1[window] = 1. * eA * nO1 * uz * np.real(cold_flux_function(Vwindows[window], Vwindows[window + 1], TeV, 1., 16., uz)) * pims_r_theta(theta)
        IO2[window] = 2. * eA * nO2 * uz * np.real(cold_flux_function(Vwindows[window], Vwindows[window + 1], TeV, 2., 16., uz)) * pims_r_theta(theta)
        IS1[window] = 1. * eA * nS1 * uz * np.real(cold_flux_function(Vwindows[window], Vwindows[window + 1], TeV, 1., 32., uz)) * pims_r_theta(theta)
        IS2[window] = 2. * eA * nS2 * uz * np.real(cold_flux_function(Vwindows[window], Vwindows[window + 1], TeV, 2., 32., uz)) * pims_r_theta(theta)
        IS3[window] = 3. * eA * nS3 * uz * np.real(cold_flux_function(Vwindows[window], Vwindows[window + 1], TeV, 3., 32., uz)) * pims_r_theta(theta)
    
    Itot = IH + IO1 + IO2 + IS1 + IS2 + IS3

    return Vwindows, Itot, IH, IO1, IO2, IS1, IS2, IS3