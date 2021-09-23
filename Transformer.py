from scipy.fftpack import dct, fft, rfft
import numpy as np
from scipy import ndimage

# ANCHOR DCT
def calculate_dct_vectors(blocks_array, offset, block_size):
    dct_array = []

    for z in range(len(blocks_array)):
        if(len(blocks_array[z]) == block_size):
            med = np.float32(dct(dct(blocks_array[z], type=3, axis=0), type=3, axis=1).flatten())
            dct_array.append(med)
            
    if offset is None:
        return dct_array
    else:
        return np.array(dct_array)[:, :offset]

def calculate_dct_blocks(blocks_array):
    dct_array = []
    for z in range(len(blocks_array)):
        dct_array.append(np.array(dct(dct(blocks_array[z], type=3, axis=0), type=3, axis=1)))

    return np.float32(np.array(dct_array))

# ANCHOR FFT
def calculate_fft_vectors(blocks_array, offset):
    fft_array = []
    for z in range(len(blocks_array)):
        fft_array.append(np.array(fft(fft(blocks_array[z], axis=0), axis=1)).flatten())

    if offset is None:
        return np.array(fft_array)
    else:
        return np.array(fft_array)[:, :offset]

def calculate_fft_blocks(blocks_array):
    fft_array = []
    for z in range(len(blocks_array)):
        fft_array.append(fft(fft(blocks_array[z], axis=0), axis=1))

    return fft_array

# ANCHOR RFFT
def calculate_rfft_vectors(blocks_array, offset, block_size):
    fft_array = []
    for z in range(len(blocks_array)):
        if(len(blocks_array[z]) == block_size):
            med = np.float32(rfft(rfft(blocks_array[z], axis=0), axis=1)).flatten()
            fft_array.append(med)

    if offset is None:
        return np.array(fft_array)
    else:
        return np.array(fft_array)[:, :offset]


def calculate_rfft_blocks(blocks_array):
    fft_array = []
    for z in range(len(blocks_array)):
        fft_array.append(np.array(rfft(rfft(blocks_array[z], axis=0), axis=1)))

    return np.array(fft_array)

# ANCHOR Gaussian filter
def apply_gaussian_filter(data, sigma):
    filtered_data = []
    for z in range(len(data)):
        filtered_data.append(ndimage.gaussian_filter(data[z], sigma))

    return np.array(filtered_data)


def apply_gaussian_filter_flatten(data, sigma, offset):
    filtered_data = []
    for z in range(len(data)):
        filtered_data.append(np.array(ndimage.gaussian_filter(data[z], sigma)).flatten())

    if offset is None:
        return np.array(filtered_data)
    else:
        return np.array(filtered_data)[:, :offset]
