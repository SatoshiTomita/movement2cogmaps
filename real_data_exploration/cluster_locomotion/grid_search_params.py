import numpy as np

# RatInABox parameters grid search space

SPEED_MEAN = np.round(np.logspace(-5.08, -3.28, 11, base=2), 5)

SPEED_CT = np.round(np.linspace(0.7, 2, 3), 3)

ROT_SPEED_STD = np.round([np.pi/6, np.pi/4, np.pi/3, 2*np.pi/4, 2*np.pi/3, 2.5*np.pi/3, 6.5*np.pi/7], 4)

THIGMOTAXIS = np.round(np.linspace(0.4, 0.6, 3), 3)
