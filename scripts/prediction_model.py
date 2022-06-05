# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 21:44:34 2021

@author: Fazuximy
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# Initiating contants for paths
path = "Z:\\python_stuff\\titanic\\titanic\\"
dataset_path = path+"datasets\\"

# Importing the training data
training_data = pd.read_csv(dataset_path+"train.csv")

test_data = pd.read_csv(dataset_path+"test.csv")
test_data_survived = pd.read_csv(dataset_path+"gender_submission.csv")

simple_predicted_survival = [0 if i == "male" else 1 for i in training_data["Sex"]]

training_data["Survived"].value_counts()/len(training_data["Survived"])*100

print(confusion_matrix(list(training_data["Survived"]),simple_predicted_survival))
print(classification_report(list(training_data["Survived"]),simple_predicted_survival))


np.array(training_data[["Pclass","Sex","Fare","Embarked"]])

np.array(training_data["Survived"])
