"""
Created on Mon Aug 12 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Analyse SAXS data.

Methods:
    transform:
        reconstruct surface profile from intensity
    deconvolve:
        correction for optical system
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
        profile = profile/np.max(profile)
        profile = np.roll(profile, int(len(profile)/2))
        profile = deconvolve(profile)
        plt.plot(range(len(profile)), profile)
        plt.xlabel("x [μm]")
        plt.ylabel("Height [μm]")
        return profile

    if len(np.shape(arr)) == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("Reconstructed surface")
        profile = np.abs(np.fft.ifft2(arr))
        profile = profile/np.max(profile)
        profile = np.roll(profile, int(len(profile)/2), 0)
        profile = np.roll(profile, int(len(profile[0])/2), 1)
        profile = deconvolve(profile)
        ax1.set_title("1-d surface")
        ax1.plot(range(len(profile[int(len(profile)/2)])),
                 profile[int(len(profile)/2)])
        ax1.set_xlabel("x [μm]")
        ax1.set_ylabel("Height [μm]")
        ax2.set_title("2-d surface")
        img = ax2.imshow(profile)
        ax2.set_xlabel("x [μm]")
        ax2.set_ylabel("y [μm]")
        plt.colorbar(img, ax=ax2, label="Height [μm]")
        fig.tight_layout()
        return profile
    raise ValueError("Input array needs to be one- or two-dimensional.")

def deconvolve(arr: np.ndarray) -> np.ndarray:
    '''
    Deconvolve one or two dimensional array

    Args:
        arr: input array

    Returns:
        devonvolved
    '''
    if len(np.shape(arr)) == 1:
        for i, val in enumerate(arr):
            x = i-len(arr)
            arr[i] = val/np.sinc(x/50)
        return arr
    for i, row in enumerate(arr):
        x = i-len(arr)/2
        for j, val in enumerate(row):
            y = j-len(row)/2
            decon = np.sinc(np.linalg.norm([x, y])/50)
            if not np.isclose(decon, 0, atol=5e-2):
                arr[i, j] = val/decon
    return arr

if __name__ == '__main__':
    data = np.loadtxt('data.txt')
    pattern = transform(data)
    plt.show()
