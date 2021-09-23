import pickle
import return_data
import keras

# ANCHOR Set parameters
n_bits          = 1                             # [1, 2, 5, 10, 20, 35, 50]
block_size      = 220                           # Mejor resultado: 220x220 en NV y 240x240 en RF
transform_type  = 'dft'                         # [dft, dct]
model_name      = './Models/cnn.h5.pkl'         # ['NaiveBayes.pkl', 'RandomForest.pkl, cnn.h5']
method          = 'reversed'                    # ['reversed', 'displaced']

if n_bits == 0:
    name = ''
else:
    name = '_' + str(n_bits) + '_' + method

parity_modified_matrix_path = 'Matlab/matrix_generated/Mat_LCP_Parity' + name + '.mat'  # Modified Matrix   [_n_bits_reversed]
random_matrix_path          = 'Matlab/matrix_generated/Mat_LCP_Urand2.mat'              # R -> True random

parity_modified_matrix_name = 'Mat_coef_recortada'
random_matrix_name          = 'Mat_coef_recortada'

print('Model name: ', model_name, '\nBlock size: ', block_size, '\nTransform type: ',
        transform_type, '\nMethod: ', method, '\nNº Bits: ', n_bits)

# ANCHOR get X, y from modified matrix
X, y = return_data.return_Xy(parity_modified_matrix_path, parity_modified_matrix_name,
                         random_matrix_path, random_matrix_name, block_size, transform_type)

# ANCHOR load model
with open(model_name, 'rb') as fid:
    model = pickle.load(fid)

# ANCHOR evaluate model
print('Accuracy of Model classifier: {:.4f}%'
     .format(100*model.score(X, y)))