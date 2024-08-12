"""
Created on Mon Aug 12 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Simulate SAXS from surface profile.
"""
import numpy as np
import matplotlib.pyplot as plt
from SS_SurfaceGen import rough, sinusoidal, crack, blob

def saxs(profile: np.ndarray) -> np.ndarray:
    '''
    Simulate SAXS profile

    Args:
        profile: surface profile

    Returns:
        SAXS detector image
    '''
    xs_img = np.fft.fft(profile)
    return np.abs(xs_img)

if __name__ == "__main__":
    # pattern = rough(100, 1, 1000)
    # pattern = sinusoidal(100, 1, 0.1)
    # pattern = crack(1000, 1, 0, 40)
    pattern = blob(1000, 1, 200, 20)
    img = saxs(pattern)
    plt.plot(range(len(img)), img)
    plt.show()
