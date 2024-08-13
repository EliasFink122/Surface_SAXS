"""
Created on Mon Aug 12 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Analyse SAXS data.


"""
import numpy as np
import matplotlib.pyplot as plt

def transform(arr: np.ndarray) -> np.ndarray:
    '''
    Transform detector read in to surface profile

    Args:
        arr: measured data

    Returns:
        surface profile

    Raises:
        ValueError: if input array does not have correct dimensions
    '''
    arr = np.sqrt(arr) # convert intensity to amplitude

    plt.figure()
    plt.title("Reconstructed surface")

    if len(np.shape(arr)) == 1:
        profile = np.abs(np.fft.fft(arr))
        profile = np.roll(profile, int(len(profile)/2))
        profile = profile/np.max(profile)
        plt.plot(range(len(profile)), profile)
        plt.xlabel("x [μm]")
        plt.ylabel("Height [μm]")
        return profile
    if len(np.shape(arr)) == 2:
        profile = np.abs(np.fft.fft2(arr))
        profile = profile/np.max(profile)
        profile = np.roll(profile, int(len(profile)/2), 0)
        profile = np.roll(profile, int(len(profile[0])/2), 1)
        plt.imshow(profile)
        plt.xlabel("x [μm]")
        plt.ylabel("y [μm]")
        plt.colorbar(label="Height [μm]")
        return profile
    raise ValueError("Input array needs to be one- or two-dimensional.")

if __name__ == '__main__':
    data = np.loadtxt('data.txt')
    pattern = transform(data)
    plt.show()
