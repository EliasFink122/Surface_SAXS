"""
Created on Mon Aug 12 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Analyse SAXS data.

Methods:
    transform:
        reconstruct surface profile from intensity
    deconvolve:
        correction for optical system
    compare:
        plot actual and reconstructed surfaces next to each other
"""
import numpy as np
import matplotlib.pyplot as plt
from SS_SurfaceGen import rough, sinusoidal, crack, blob, laser
from SS_Simulation import saxs, saxs2d

def transform(arr: np.ndarray, plot = True) -> np.ndarray:
    '''
    Transform detector read in to surface profile

    Args:
        arr: measured data

    Returns:
        surface profile

    Raises:
        ValueError: if input array does not have correct dimensions
    '''
    arr = np.sqrt(arr) # intensity -> amplitude

    if len(np.shape(arr)) == 1:
        plt.figure()
        plt.title("Reconstructed surface")
        profile = np.abs(np.fft.ifft(arr))
        profile = np.roll(profile, int(len(profile)/2))
        # profile = deconvolve(profile)
        profile = profile/np.max(profile)
        plt.plot(range(len(profile)), profile)
        plt.xlabel("x [μm]")
        plt.ylabel("Height [μm]")
        if not plot:
            plt.close()
        return profile

    if len(np.shape(arr)) == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("Reconstructed surface")
        profile = np.abs(np.fft.ifft2(arr))
        profile = np.roll(profile, int(len(profile)/2), 0)
        profile = np.roll(profile, int(len(profile[0])/2), 1)
        # profile = deconvolve(profile)
        profile = profile/np.max(profile)
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
        if not plot:
            plt.close()
        return profile
    raise ValueError("Input array needs to be one- or two-dimensional.")

def deconvolve(arr: np.ndarray) -> np.ndarray:
    '''
    Deconvolve one or two dimensional array

    Args:
        arr: input signal

    Returns:
        devonvolved signal
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

def compare(num: int):
    '''
    Compare actual surface with reconstructed

    Args:
        num: number of surface elements
    '''
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle("Comparison between surfaces")

    profile = laser(num, amp=1, width=num/10)
    surf1 = ax1.imshow(profile)
    ax1.set_title("Surface profile")
    ax1.set_xlabel("x [μm]")
    ax1.set_ylabel("y [μm]")
    plt.colorbar(surf1, ax=ax1)

    img = saxs2d(profile, False)

    reconstructed = transform(img)
    surf2 = ax2.imshow(reconstructed)
    ax2.set_title("Reconstructed profile")
    ax2.set_xlabel("x [μm]")
    ax2.set_ylabel("y [μm]")
    plt.colorbar(surf2, ax=ax2)

if __name__ == '__main__':
    # data = np.loadtxt('data.txt')
    # pattern = transform(data)
    compare(1000)
    plt.show()
