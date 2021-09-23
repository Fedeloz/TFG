from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate
import pickle
import return_data

# ANCHOR Read data
parity_matrix_path = 'Matlab/matrix_generated/Mat_LCP_Parity.mat'   # L -> Pseudo random
random_matrix_path = 'Matlab/matrix_generated/Mat_LCP_Urand2.mat'   # R -> True random

# parity_matrix_name = 'Matrix'
parity_matrix_name = 'Mat_coef_recortada'
random_matrix_name = 'Mat_coef_recortada'

block_size      = 220       # Mejor resultado: 220x220
transform_type  = 'dft'     # dft, dct

X, y = return_data.return_Xy(parity_matrix_path, parity_matrix_name, random_matrix_path, random_matrix_name, block_size, transform_type)

scoring = ['accuracy', 'f1_micro', 'roc_auc', 'recall']

# ANCHOR Build model
print("Starting to train")
naiveBayes = GaussianNB()

scores = cross_validate(naiveBayes, X, y, cv=10,scoring=scoring, return_train_score=True, verbose=3)
# cv -> splitting straegy, verbosity -> how much information do you want
print(scores)

print("Accuracy: %.2f%%" % (scores['test_accuracy'].mean()*100.0))
print("F1: %.2f%%" % (scores['test_f1_micro'].mean()*100.0))
print("ROC: %.2f%%" % (scores['test_roc_auc'].mean()*100.0))
print("Recall: %.2f%%" % (scores['test_recall'].mean()*100.0))
print("Fit time: %.2f%%" % (scores['fit_time'].mean()))

print('\nSaving model as "naive_bayes.pkl"...')

# ANCHOR save model
with open('./Models/NaiveBayes.pkl', 'wb') as f:
    pickle.dump(naiveBayes.fit(X, y), f)

print('Model saved!')