"""
Created on Mon Aug 12 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Analyse SAXS data.


"""
import numpy as np
import matplotlib.pyplot as plt

def transform(arr: np.nadrray) -> np.ndarray:
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
        profile = np.fft.fft(arr)
        plt.plot(range(len(profile)), profile)
        return profile
    if len(np.shape(arr)) == 2:
        profile = np.fft.fft2(arr)
        plt.imshow(profile)
        return profile
    raise ValueError("Input array needs to be one- or two-dimensional.")

if __name__ == '__main__':
    data = np.loadtxt('data')
    pattern = transform(data)
