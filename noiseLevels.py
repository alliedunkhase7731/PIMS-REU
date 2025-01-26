import csv
import numpy as np
from conversions import eV2kms, ZV2kms
from sc_to_fc import SC_to_FC
from cold_flux_function import cold_flux_function
from calculate_currentsNoise import calculate_currents
import matplotlib.pyplot as plt

# Constants, ensuring they are defined in your script or imported if they're in another module
e0_pC = 1.6021892e-7  # elementary charge in picocoulombs
cm2km = 1e5  # conversion from centimeters to kilometers

# Initial parameter values
nwin_0 = 64
Vlow_0 = 10.
Vhigh_0 = 8000.
alpha_0 = 180
umag_0 = 100.
TeV_0 = 100.
necm3_0 = 25.
Xp_0 = 0.1
OSratio_0 = 2.
ZO2_0 = .75
ZS2_0 = 0.6
ZS3_0 = 0.2

# Noise levels
noise_levels = [0.033, 0.1, 0.33, 1.0, 3.33]

# Calculate theta, uz, phi for all Faraday cups
thetas, uzs, phis = SC_to_FC(alpha_0, umag_0)

# Select the first Faraday cup's parameters
theta = thetas[0]
uz = uzs[0]
phi = phis[0]

# Create plots
fig, axs = plt.subplots(1, len(noise_levels), figsize=(15, 3), sharey=True)
for i, noise_level in enumerate(noise_levels):
    Vwindows, Itot_noise, _ = calculate_currents(nwin_0, Vlow_0, Vhigh_0, theta, uz, TeV_0, necm3_0, Xp_0, OSratio_0, ZO2_0, ZS2_0, ZS3_0, phi, noise_level)
    axs[i].plot(Vwindows, Itot_noise, label=f'Noise: {noise_level} pA', color='blue')
    axs[i].set_title(f'Noise level: {noise_level} pA')
    axs[i].set_xscale('log')
    axs[i].set_xlabel('Voltage (V)')
    axs[i].grid(True)
    if i == 0:
        axs[i].set_ylabel('Current (pA)')

plt.legend()
plt.tight_layout()
plt.show()