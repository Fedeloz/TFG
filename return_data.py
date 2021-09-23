import numpy as np
import logging as log
import matplotlib.pyplot as plt
# import scipy.io
import mat73
import Transformer as tr
import pandas as pd
from sklearn import preprocessing


# Given path, name, blocksize and transform typ returns X and y
def return_Xy(parity_matrix_path, parity_matrix_name, random_matrix_path, random_matrix_name, block_size, transform_type):

    # Get blocks -> parity and random matrix
    parity_blocks = get_matrix_blocks(parity_matrix_path, parity_matrix_name, block_size)
    random_blocks = get_matrix_blocks(random_matrix_path, random_matrix_name, block_size)
    print("Blocks calculated")

    # Vector transform -> parity and random blocks
    offset = None
    matrix_tr, random_tr = calculate_tranform_gateway(transform_type, parity_blocks, random_blocks, offset, block_size)
    print("Blocks transformed")

    # Labeled DataFrame
    print("Creating pandas dataframe")
    dataset = create_dataset(matrix_tr, random_tr)
    df = pd.DataFrame(np.float32(dataset))
    du = pd.get_dummies(df)
    original = du.values

    # X, y
    y = []
    NUMBER_OF_FEATURES_TO_USE = len(original) # You can modify this parameter to run the model with less features
    for i in range(len(original)):
        y.append(original[i][len(du.columns) - 1])

    X = np.array(preprocessing.normalize(
        np.delete(original, len(du.columns) - 1, axis=1)))[:, :NUMBER_OF_FEATURES_TO_USE]  # The labels are removed, here are stored all the features

    print("X and y variables created")

    return X, y

# Creates Dataset
def create_dataset(parity_data, random_data):
    parity_dataset = np.float32(np.array(add_label(parity_data, 0)))    # Parity are labeled with 0
    random_dataset = np.float32(np.array(add_label(random_data, 1)))    # Random are labeled with 1

    dataset = np.concatenate((parity_dataset, random_dataset))          # Concatenate both
    log.info('Dataset lenght: %d', len(dataset))
    log.info('Dataset number of features: %d', len(dataset[0]))
    return dataset

# Adds label to an array
def add_label(array, label):
    array_result = []
    for t in range(len(array)):
        array_result.append(np.append(array[t], label))

    return array_result

# Transform random and parity blocks into vectors
def calculate_tranform_gateway(transform_type, parity_blocks, random_blocks, offset, block_size):
    log.info(' Calculating tranform for: %s', transform_type)
    if transform_type == 'dct':
        return get_dct_vectors(parity_blocks, random_blocks, offset, block_size)
    elif transform_type == 'dft':
        return get_dft_vectors(parity_blocks, random_blocks, offset, block_size)
    elif transform_type == 'dct_blur':
        return get_dct_blur_vectors(parity_blocks, random_blocks, offset)
    elif transform_type == 'dft_blur':
        return get_dft_blur_vectors(parity_blocks, random_blocks, offset)
    elif transform_type == 'normal':
        return get_normal_vectors(parity_blocks, random_blocks)

# Returns blocks of matrix concatenated with step step
def extract_blocks(step, matrix):
    blocks = []
    width, height = matrix.shape
    i = 0
    for y in range(0, height, step):
        for x in range(0, width, step):
            block = matrix[y:y + step, x:x + step]
            i += 1
            if(block.shape == (step, step)): #Just the perfect matrix are appended
                blocks.append(block)

    return blocks

# Get matrix blocks
def get_matrix_blocks(path, name, block_size):                      
    log.debug('Extracting matrix blocks for: %s', path)      # track events
    extracted_matrix = load(path, name)                      # load matrix
    blocks = extract_blocks(block_size, extracted_matrix)    # extract blocks
    return blocks

# Returns matrix loaded
def load(ruta, name):
    mat = mat73.loadmat(ruta)
    # mat = scipy.io.loadmat(ruta)  # deprecated
    matrix = mat[name].astype(np.float)
    width, height = matrix.shape
    return matrix


# Returns one dimension array
def get_normal(blocks):
    array = []
    for z in range(len(blocks)):
        array.append(blocks[z].flatten())   # The array is collapsed into one dimension
    return np.array(array)

# Returns one dimension blocks
def get_normal_vectors(parity_blocks, random_blocks):
    parity = get_normal(parity_blocks)
    random = get_normal(random_blocks)

    return parity, random

# DCT vectors
def get_dct_vectors(parity_blocks, random_blocks, offset, block_size):
    parity_dct = tr.calculate_dct_vectors(parity_blocks, offset, block_size)
    random_dct = tr.calculate_dct_vectors(random_blocks, offset, block_size)

    return parity_dct, random_dct

# DFT vectors
def get_dft_vectors(parity_blocks, random_blocks, offset, block_size):
    parity_fft = tr.calculate_rfft_vectors(parity_blocks, offset, block_size)
    random_fft = tr.calculate_rfft_vectors(random_blocks, offset, block_size)

    return parity_fft, random_fft

# DCT blur vectors
def get_dct_blur_vectors(parity_blocks, random_blocks, offset):
    parity_dct_blocks = tr.calculate_dct_blocks(parity_blocks)
    random_dct_blocks = tr.calculate_dct_blocks(random_blocks)

    parity_dct_blur = tr.apply_gaussian_filter_flatten(parity_dct_blocks, 4, offset)
    random_dct_blur = tr.apply_gaussian_filter_flatten(random_dct_blocks, 4, offset)

    return parity_dct_blur, random_dct_blur

# DFT blur vectors
def get_dft_blur_vectors(parity_blocks, random_blocks, offset):
    parity_dft_blocks = tr.calculate_rfft_blocks(parity_blocks)
    random_dft_blocks = tr.calculate_rfft_blocks(random_blocks)

    parity_dft_blur = tr.apply_gaussian_filter_flatten(parity_dft_blocks, 4, offset)
    random_dft_blur = tr.apply_gaussian_filter_flatten(random_dft_blocks, 4, offset)

    return parity_dft_blur, random_dft_blur
