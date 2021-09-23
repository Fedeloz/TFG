import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
import pickle
import return_data

logging.basicConfig(level=logging.INFO)     # Tool to prevent errors and attacks and have analysis

# ANCHOR Read data
parity_matrix_path = 'Matlab/matrix_generated/Mat_LCP_Parity.mat'
random_matrix_path = 'Matlab/matrix_generated/Mat_LCP_Urand2.mat'

parity_matrix_name = 'Mat_coef_recortada'
random_matrix_name = 'Mat_coef_recortada'

block_size      = 240
transform_type  = 'dft'

X, y = return_data.return_Xy(parity_matrix_path, parity_matrix_name, random_matrix_path, random_matrix_name, block_size, transform_type)

scoring = ['accuracy', 'f1_micro', 'roc_auc', 'recall']

# ANCHOR Build model
print("Starting to train")
rfc = RandomForestClassifier(bootstrap=False, class_weight="balanced_subsample", criterion="gini", max_depth=50,
                             n_estimators=1000, n_jobs=3)


scores = cross_validate(rfc, X, y, cv=10,scoring=scoring,return_train_score=True, verbose=3)

print(scores)

print("Accuracy: %.2f%%" % (scores['test_accuracy'].mean()*100.0))
print("F1: %.2f%%" % (scores['test_f1_micro'].mean()*100.0))
print("ROC: %.2f%%" % (scores['test_roc_auc'].mean()*100.0))
print("Recall: %.2f%%" % (scores['test_recall'].mean()*100.0))
print("Fit time: %.2f%%" % (scores['fit_time'].mean()))

# ANCHOR save model
with open('./Models/RandomForest.pkl', 'wb') as f:
    pickle.dump(rfc.fit(X, y), f)

print('Model saved!')