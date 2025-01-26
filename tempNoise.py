# Change code for spectra of JUST hydrogen

import pandas as pd
import csv
import numpy as np
from conversions import eV2kms, ZV2kms
from sc_to_fc import SC_to_FC
from cold_flux_function import cold_flux_function
from calculate_currentsNoise import calculate_currents
import matplotlib.pyplot as plt

# Constants and fixed parameters
fixed_parameters = {
    'nwin_0': 64,
    'Vlow_0': 10.,
    'Vhigh_0': 8000.,
    'necm3_0': 25
}
alpha_0 = 90
umag_0 = 100.

# Noise levels
noise_levels = [0.033, 0.1, 0.33, 1.0, 3.33]
simulations = 1000  # number of simulations

# Create dynamic parameter distributions
parameters_list = []
temperatures = []  # To store temperatures for each simulation
for _ in range(simulations):
    parameters = {
        'uz_0': np.random.uniform(50, 150),
        'TeV_0': np.random.uniform(50, 150),
        'Xp_0': np.random.uniform(0.05, 0.15),
        'OSratio_0': np.random.uniform(1.5, 2.5),
        'ZO2_0': np.random.uniform(0.5, 1.0),
        'ZS2_0': np.random.uniform(0.4, 0.8),
        'ZS3_0': np.random.uniform(0.1, 0.3)
    }
    parameters_list.append(parameters)
    temperatures.append(parameters['TeV_0'])  # Store temperature

# Calculate theta, uz, phi for all Faraday cups
thetas, uzs, phis = SC_to_FC(alpha_0, umag_0)
theta = thetas[0]
phi = phis[0]

# Prepare data storage
all_currents = []
all_temperatures = []

# Run simulations for each noise level
for noise_level in noise_levels:
    currents_for_noise = []
    temperatures_for_noise = []
    for params in parameters_list:
        Vwindows, Itot_noise, rho_total = calculate_currents(fixed_parameters['nwin_0'], fixed_parameters['Vlow_0'], fixed_parameters['Vhigh_0'], 
                                                             theta, params['uz_0'], params['TeV_0'], fixed_parameters['necm3_0'], 
                                                             params['Xp_0'], params['OSratio_0'], params['ZO2_0'], params['ZS2_0'], params['ZS3_0'], phi, noise_level)
        currents_for_noise.append(Itot_noise)
        temperatures_for_noise.append(params['TeV_0'])
    all_currents.append(currents_for_noise)
    all_temperatures.append(temperatures_for_noise)

# Plotting the first simulation of each noise level
fig, axs = plt.subplots(1, len(noise_levels), figsize=(15, 3), sharey=True)
for i, noise_level in enumerate(noise_levels):
    axs[i].plot(Vwindows, all_currents[i][0], label=f'Noise: {noise_level} pA', color='blue')
    axs[i].set_title(f'Noise level: {noise_level} pA')
    axs[i].set_xscale('log')
    axs[i].set_xlabel('Voltage (V)')
    axs[i].grid(True)
    if i == 0:
        axs[i].set_ylabel('Current (pA)')

plt.legend()
plt.tight_layout()
plt.show()

# Saving results to CSV
for i, noise_level in enumerate(noise_levels):
    df_currents = pd.DataFrame(all_currents[i], columns=[f'Voltage_{v:.2f}' for v in Vwindows])
    df_currents.to_csv(f'currents_temp_noise_{noise_level}.csv', index=False)

    df_temperatures = pd.DataFrame({'temperature': all_temperatures[i]})
    df_temperatures.to_csv(f'temp_with_noise_{noise_level}.csv', index=False)

print("CSV files for currents and temperatures have been saved successfully.")