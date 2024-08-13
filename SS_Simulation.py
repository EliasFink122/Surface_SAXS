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

def saxs(profile: np.ndarray) -> np.ndarray:
    '''
    Simulate SAXS profile

    Args:
        profile: surface profile

    Returns:
        SAXS detector image

    Raises:
        ValueError: if input array is not one dimensional
    '''
    if len(np.shape(profile)) != 1:
        raise ValueError("This function only works for 1d array. Use saxs2d() for 2d.")
    xs_img = np.fft.fft(profile)
    xs_img = np.roll(xs_img, int(len(xs_img)/2))
    plt.figure()
    plt.title("SAXS image")
    plt.xlabel("x [mm]")
    plt.ylabel("Intensity [A.U.]")
    plt.plot(range(len(xs_img)), xs_img)
    return np.square(np.abs(xs_img))

def saxs2d(profile, log = False) -> np.ndarray:
    '''
    Simulate SAXS profile in 2d

    Args:
        profile: surface profile

    Returns:
        SAXS detector image
    '''
    if len(np.shape(profile)) == 1:
        profile = make2d(profile)
    xs_img = np.fft.fft2(profile)
    xs_img = np.roll(xs_img, int(len(xs_img)/2), 0)
    xs_img = np.roll(xs_img, int(len(xs_img[0])/2), 1)

    plt.figure()
    plt.title("SAXS image")
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    if log:
        plt.imshow(np.log(np.abs(xs_img)))
    else:
        plt.imshow(np.square(np.abs(xs_img)))
    plt.colorbar(label="Intensity [A.U.]")
    return np.square(np.abs(xs_img))

if __name__ == "__main__":
    # pattern = rough(100, 1, 1000)
    # pattern = sinusoidal(100, 1, 0.1)
    # pattern = crack(1000, 1, 0, 40)
    # pattern = blob(1000, 1, 200, 20)
    pattern = laser(1000, 1, 50)
    img = saxs2d(pattern)
    save(img)
    plt.show()
