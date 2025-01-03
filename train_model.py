import pandas as pd
from find_motifs import *
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import average_precision_score
from imblearn.under_sampling import RandomUnderSampler
import random
import numpy as np

np.random.seed(0)
random.seed(0)

# Function to undersample the dataset to balance positive and negative samples
def under_sample(x_train, y_train, ratio=1):
    rus = RandomUnderSampler(sampling_strategy=ratio, random_state=0)
    x_resampled, y_resampled = rus.fit_resample(x_train, y_train)
    return x_resampled, y_resampled

# Function to read the training and testing data from pickle files
def read_data(dataset, s):
    with open('split_dataset/' + dataset + '/' + s + '_mean.pickle', 'rb') as f:
        x = pickle.load(f)
    x = x.apply(lambda a: (a - a.min()) / (a.max() - a.min()))  # Normalize data

    with open('processing_dataset/' + dataset + '/y_' + s + '.pickle', 'rb') as f:
        y = pickle.load(f)

    return x, y


if __name__ == "__main__":
    dataset_list = ['email-Enron', 'email-Eu', 'NDC-classes', 'NDC-substances', 'contact-primary-school', 
                    'contact-high-school', 'coauth-MAG-History', 'DAWN', 'threads-ask-ubuntu', 'tags-ask-ubuntu']

    # Iterate through each dataset
    for dataset in dataset_list:
        print(dataset)
        # Read training and testing data
        x_train, y_train = read_data(dataset, 'train')
        x_test, y_test = read_data(dataset, 'test')

        rb = sum(y_test) / len(y_test)  # random baseline

        # Apply undersampling to balance classes in training data
        x_train, y_train = under_sample(x_train, y_train, ratio=0.33)

        # Train the model (Logistic Regression in this case)
        model = LogisticRegression(solver='liblinear', penalty='l2', max_iter=1000)
        model.fit(x_train, y_train)

        # Predictions
        y_pre = model.predict_proba(x_test)[:, 1]
        ap = average_precision_score(y_test, y_pre)
        pm = ap / rb

        print('pm', round(ap / rb, 3), 'ap', ap)