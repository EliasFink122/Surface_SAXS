"""
Created on Mon Aug 12 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Generate surface profile for SAXS simulation.

Methods:
    rough:
        random rough surface pattern
    sinusoidal:
        periodic, smooth surface pattern
    crack:
        even surface with big crack
    blob:
        super Gaussian shape blobs on surface
    make2d:
        tensor product of array with itself
"""
import numpy as np
import matplotlib.pyplot as plt

def rough(num: int, amp: float, smooth = 0) -> np.ndarray:
    '''
    Rough surface pattern

    Args:
        num: number of surface height values
        amp: amplitude of surface roughness
        smooth: number of smoothings
    
    Returns:
        surface profile
    '''
    surface = amp*np.random.rand(num)
    if smooth == 0:
        return surface
    smoothed_surface = surface.copy()
    for _ in range(smooth):
        for j, val in enumerate(surface):
            if j in (0, num-1):
                continue
            smoothed_surface[j] = (surface[j-1] + 2*val + surface[j+1])/4
    return smoothed_surface

def sinusoidal(num: int, amp: float, freq: float) -> np.ndarray:
    '''
    Sinusoidal surface pattern

    Args:
        num: number of surface height values
        amp: amplitude of surface modulation
        freq: frequency of surface modulation
    
    Returns:
        surface profile
    '''
    surface = np.zeros(num)
    for i, _ in enumerate(surface):
        surface[i] = amp*np.sin(freq*i)
    return surface

def crack(num: int, amp: float, pos: float, width: float) -> np.ndarray:
    '''
    Cracked surface pattern

    Args:
        num: number of surface height values
        amp: amplitude of surface modulation
        pos: centre of crack
        width: width of crack
    
    Returns:
        surface profile
    '''
    surface = np.zeros(num)
    for i, _ in enumerate(surface):
        x = i - num/2
        surface[i] = np.min([0, amp*(2*np.abs(x-pos)/width-1)])
    return surface

def blob(num: int, amp: float, spacing: float, width: float) -> np.ndarray:
    '''
    Blobs surface pattern

    Args:
        num: number of surface height values
        amp: amplitude of surface modulation
        spacing: spacing between blobs
        width: width of crack
    
    Returns:
        surface profile
    '''
    surface = np.zeros(num)
    for i, _ in enumerate(surface):
        x = i - num/2
        surface[i] = amp*np.sum([np.add(np.exp(-((x-j*spacing)/(2*width))**4),
                                        np.exp(-((x+(j+1)*spacing)/(2*width))**4))
                                for j in range(round(num/spacing))])
    return surface

def laser(num: int, amp: float, width: float) -> np.ndarray:
    '''
    Laser imprint pattern

    Args:
        num: number of surface height values
        amp: amplitude of laser imprint
        width: width of laser
    
    Returns:
        surface profile
    '''
    surface = np.zeros((num, num))
    for i, row in enumerate(surface):
        for j, _ in enumerate(row):
            x = i - num/2
            y = j - num/2
            surface[i, j] = amp * np.exp(-((x**2 + y**2)/(2*width)**2)**5)
    return surface

def make2d(arr: np.ndarray) -> np.ndarray:
    '''
    Take tensor product of array with itself

    Args:
        arr: 1d numpy array

    Returns:
        2d array
    '''
    return np.tensordot(arr, arr, 0)

def expand2d(arr: np.ndarray) -> np.ndarray:
    '''
    Expand array to 2d

    Args:
        arr: 1d numpy array

    Returns:
        2d array
    '''
    new_arr = [arr for _ in arr]
    return np.array(new_arr)

def save(arr: np.ndarray, fname = ""):
    '''
    Save data in txt file

    Args:
        arr: array to save
        fname: file name
    '''
    np.savetxt('data' + fname + '.txt', arr)

if __name__ == "__main__":
    # pattern = rough(100, 1, 1000)
    # pattern = sinusoidal(100, 1, 0.1)
    # pattern = crack(100, 1, 0, 40)
    pattern = blob(1000, 1, 200, 20)
    plt.title("Surface profile")
    plt.imshow(make2d(pattern))
    plt.xlabel("x [μm]")
    plt.ylabel("y [μm]")
    plt.colorbar(label="Height [μm]")
    # plt.plot(range(len(pattern)), pattern)
    plt.show()
