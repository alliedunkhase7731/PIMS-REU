import numpy as np

def SC_to_FC(alpha, umag):
    # Define rotation matrices for each Faraday cup
    R1 = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]])

    R2 = np.array([[1, 0, 0],
                   [0, 0, -1],
                   [0, 1, 0]])

    R3 = np.array([[1, 0, 0],
                   [0, -1, 0],
                   [0, 0, -1]])

    R4 = np.array([[1, 0, 0],
                   [0, 0, 1],
                   [0, -1, 0]])

    # Calculate the velocity vector in the spacecraft coordinate system
    cos_alpha = np.cos(np.radians(alpha))
    sin_alpha = np.sin(np.radians(alpha))
    u_SC = umag * np.array([0, sin_alpha, -cos_alpha])

    # Apply rotation matrices to the velocity vector to get velocities with respect to Faraday Cups
    u_FC1 = R1 @ u_SC
    u_FC2 = R2 @ u_SC
    u_FC3 = R3 @ u_SC
    u_FC4 = R4 @ u_SC

    # Calculate uz (z_FC) for each Faraday cup
    uz_FC1 = u_FC1[2]
    uz_FC2 = u_FC2[2]
    uz_FC3 = u_FC3[2]
    uz_FC4 = u_FC4[2]

    def calculate_theta(u_FC):
        x_FC, y_FC, z_FC = u_FC
        return np.degrees(np.arccos(z_FC / np.sqrt(x_FC**2 + y_FC**2 + z_FC**2)))

    def calculate_phi(u_FC):
        x_FC, y_FC, z_FC = u_FC
        return np.degrees(np.arctan(x_FC / y_FC))

    # Calculate theta for each Faraday cup
    theta_FC1 = calculate_theta(u_FC1)
    theta_FC2 = calculate_theta(u_FC2)
    theta_FC3 = calculate_theta(u_FC3)
    theta_FC4 = calculate_theta(u_FC4)

    # Calculate phi for each Faraday cup
    phi_FC1 = calculate_phi(u_FC1)
    phi_FC2 = calculate_phi(u_FC2)
    phi_FC3 = calculate_phi(u_FC3)
    phi_FC4 = calculate_phi(u_FC4)

    return [theta_FC1, theta_FC2, theta_FC3, theta_FC4], [uz_FC1, uz_FC2, uz_FC3, uz_FC4], [phi_FC1, phi_FC2, phi_FC3, phi_FC4]
