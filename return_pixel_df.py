#----------------------------------------------------------------------
#   This script will return a df with labeled parity and urand images  |
#----------------------------------------------------------------------

import scipy.io
import numpy as np
import pandas as pd
import logging as log
import mat73

# ANCHOR return DataFrame
def returnDF(parity_matrix_path, random_matrix_path, num_img, size, data= 'Data'):
    # read data
    parity      = scipy.io.loadmat(parity_matrix_path)[data]  # deprecated
    urand2      = scipy.io.loadmat(random_matrix_path)[data]

    # parity      = mat73.loadmat(parity_matrix_path)[data]
    # urand2      = mat73.loadmat(random_matrix_path)[data]
    data, label, f_row = return_data(parity, urand2, size, num_img)
    df          = pd.DataFrame(data)
    df.columns  = f_row # set the header row as the df header

    return df

# ANCHOR data
def return_data(parity, urand2, size, num_img):
    f_row = first_row(size)
    data  = []
    label = []

# Parity
    for i in range(num_img):
        row_parity   = [0]        # label parity
        for j in range(len(f_row)):
            row_parity.append(parity[0, j + len(f_row)*i])

        data.append((row_parity))

# Urand2
    for i in range(num_img):
        row_urand2   = [1]        # label random
        for j in range(len(f_row)):
            row_urand2.append(urand2[0, j + len(f_row)*i])

        data.append((row_urand2))


    f_row.insert(0, 'Label')
    return data, label, f_row

# ANCHOR first row
def first_row(size):
    first_row = []
    for i in range(size**2):
        first_row.append('Pixel ' + str(i))

    return first_row