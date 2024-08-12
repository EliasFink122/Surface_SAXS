"""
Created on Mon Aug 12 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Simulate SAXS from surface profile.

Methods:
    saxs:
        small angle xray scattering in 1d
    saxs2d:
        small angle xray scattering in 2d
"""
import numpy as np
import matplotlib.pyplot as plt
from SS_SurfaceGen import *

def saxs(profile: np.ndarray, half = True) -> np.ndarray:
    '''
    Simulate SAXS profile

    Args:
        profile: surface profile

    Returns:
        SAXS detector image
    '''
    xs_img = np.fft.fft(profile)
    if half:
        plt.plot(range(int(len(xs_img)/2)), xs_img[:int(len(xs_img)/2)])
        return np.abs(xs_img[:int(len(xs_img)/2)])
    plt.plot(range(len(xs_img)), xs_img)
    return np.abs(xs_img)

def saxs2d(profile, log = False) -> np.ndarray:
    '''
    Simulate SAXS profile in 2d

    Args:
        profile: surface profile

    Returns:
        SAXS detector image
    '''
    profile = make2d(profile)
    xs_img = np.fft.fft2(profile)
    if log:
        plt.imshow(np.log(np.abs(xs_img)))
    else:
        plt.imshow(np.abs(xs_img))
    plt.colorbar()
    return xs_img

if __name__ == "__main__":
    # pattern = rough(100, 1, 1000)
    # pattern = sinusoidal(100, 1, 0.1)
    # pattern = crack(1000, 1, 0, 40)
    pattern = blob(1000, 1, 200, 20)
    img = saxs(pattern, False)
    save(img)
    plt.show()
