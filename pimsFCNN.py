import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# List of noise levels
# fill with your respective noise levels: noise_levels = []
mse_scores = []
percent_error_mass_density = []

# Function to create the model
def create_model(input_dim):
    model = Sequential()
    model.add(Dense(8, input_dim=input_dim, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Adjust if y_train.shape[1] is not 1
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

# Loop through each noise level
for noise in noise_levels:
    # Load data
    # current_values_df = pd.read_csv(f'your file name for VDF')
   # for parameter you're tesing: parameter = pd.read_csv(f'your file name for parameter ')

    # Prepare data
    X = current_values_df.values
    y = mass_den_df.values.ravel()  # Ensure y is the correct shape

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)

    # Train model
    model = create_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=50, batch_size=20, verbose=1)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    y_pred_original = scaler_y.inverse_transform(y_pred)
    y_test_original = scaler_y.inverse_transform(y_test)

    mse = mean_squared_error(y_test_original, y_pred_original)
    mse_scores.append(mse)

    # Compute percent error in mass density
    percent_error = 100 * np.mean(np.abs((y_test_original - y_pred_original) / y_test_original))
    percent_error_parameter.append(percent_error)

    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_original, y_pred_original, alpha=0.5)
    plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--')
    plt.title(f'Actual vs Predicted Parameter at Noise {noise} (units)')
    plt.xlabel('Actual Parameter (units)')
    plt.ylabel('Predicted Total Parameter (units)')
    plt.grid(True)
    plt.show()

# Plot MSE as a function of percent error in mass density with noise level annotations
plt.figure(figsize=(10, 6))
plt.plot(percent_error_mass_density, mse_scores, marker='o', linestyle='-', color='b', label='MSE')
for i, noise in enumerate(noise_levels):
    plt.annotate(f'Noise: {noise}', (percent_error_parameter[i], mse_scores[i]), textcoords="offset points", xytext=(5,5), ha='right')
plt.title('Mean Squared Error vs. Percent Error in Parameter ((Units)²)/%')
plt.xlabel('Percent Error in Mass Density (%)')
plt.ylabel('MSE ((Units)²)')
plt.grid(True)
plt.legend()
plt.show()
