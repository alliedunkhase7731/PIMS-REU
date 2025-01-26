import numpy as np

def pims_r_theta(theta_deg):
    x = np.abs(np.tan(theta_deg * np.pi / 180.))
    f = [-0.32951795, -0.40417760]
    g = [0.36, 1.2, 2.76]
    m1 = 10.**(f[0] + f[1] * x)
    m2 = g[0] * np.exp(-(g[1] * (x ** g[2])))
    m2 = m2 + 999. * (x < 0.55)
    model = np.minimum(m1, m2)
    return model
