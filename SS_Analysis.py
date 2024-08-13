"""
Created on Mon Aug 12 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Analyse SAXS data.

Methods:
    transform:
        reconstruct surface profile from intensity
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

    if len(np.shape(arr)) == 1:
        plt.figure()
        plt.title("Reconstructed surface")
        profile = np.abs(np.fft.ifft(arr))
        profile = np.roll(profile, int(len(profile)/2))
        profile = profile/np.max(profile)
        plt.plot(range(len(profile)), profile)
        plt.xlabel("x [μm]")
        plt.ylabel("Height [μm]")
        return profile

    if len(np.shape(arr)) == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("Reconstructed surface")
        profile = np.abs(np.fft.ifft2(arr))
        profile = profile/np.max(profile)
        profile = np.roll(profile, int(len(profile[0])/2), 1)
        ax1.set_title("1-d surface")
        ax1.plot(range(len(profile[0])), profile[0])
        ax1.set_xlabel("x [μm]")
        ax1.set_ylabel("Height [μm]")
        profile = np.roll(profile, int(len(profile)/2), 0)
        ax2.set_title("2-d surface")
        img = ax2.imshow(profile)
        ax2.set_xlabel("x [μm]")
        ax2.set_ylabel("y [μm]")
        plt.colorbar(img, ax=ax2, label="Height [μm]")
        fig.tight_layout()
        return profile
    raise ValueError("Input array needs to be one- or two-dimensional.")

if __name__ == '__main__':
    data = np.loadtxt('data.txt')
    pattern = transform(data)
    plt.show()
