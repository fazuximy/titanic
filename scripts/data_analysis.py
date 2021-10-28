# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 21:43:41 2021

@author: Fazuximy
"""

# Importing libraries
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import math

# Initiating contants for paths
path = "Z:\\python_stuff\\titanic\\titanic\\"
dataset_path = path+"datasets\\"

# Importing the training data
training_data = pd.read_csv(dataset_path+"train.csv")


# survival - Survival - 0 = No, 1 = Yes
# pclass - Ticket - class 1 = 1st, 2 = 2nd, 3 = 3rd
# sex - Sex	
# Age - Age in years	
# sibsp - number of siblings / spouses aboard the Titanic	
# parch - number of parents / children aboard the Titanic	
# ticket - Ticket number	
# fare - Passenger fare	
# cabin - Cabin number	
# embarked - Port of Embarkation - C = Cherbourg, Q = Queenstown, S = Southampton


# Feature engineering
# Adding a column of the title
    # It could be interesting to discover whether a title could have an impact or indicate a confounding variable
person_title = [re.findall("\,\s([\w+|\w+\s+\w]+\.)",i)[0] if len(re.findall("\,\s([\w+|\w+\s+\w]+\.)",i)) > 0 else None for i in training_data["Name"]]
training_data["title"] = person_title
# Some of passenger fares are extremely high so it is log transformed to be able to better visualize
training_data["log_fare"] = [math.log(i) if i != 0 else 0 for i in training_data["Fare"]]

#Displaying the columns for overview
training_data.columns

# Plotting the categorical features in order to examine how the survival rate is distributed
    # These plots are the total count of passengers in each category
# The passenger class
sns.catplot(x="Pclass", col="Survived", data=training_data, ci = None, kind = "count")

# The sex of the passenger
sns.catplot(x="Sex", col="Survived", data=training_data, ci = None, kind = "count")

# The place where the passenger embarked
sns.catplot(x="Embarked", col="Survived", data=training_data, ci = None, kind = "count")


# Plotting the same categories but this time, it is the percentage of how many survied in each category
    # This is done in order to
sns.barplot(x="Pclass", y="Survived", data=training_data, ci = None)

sns.barplot(x="Sex", y="Survived", data=training_data, ci = None)

sns.barplot(x="Embarked", y="Survived", data=training_data, ci = None)

# Here the survival rate plot for the titles are added as well
sns.barplot(x="title", y="Survived", data=training_data, ci = None)
plt.xticks(rotation=90)

sns.histplot(x="Age", hue="Survived", data=training_data)

sns.histplot(x="SibSp", hue="Survived", data=training_data, bins = 8)

sns.histplot(x="Parch", hue="Survived", data=training_data, bins = 6)

sns.histplot(x="Fare", hue="Survived", data=training_data, log_scale = False)

sns.histplot(x="log_fare", hue="Survived", data=training_data, log_scale = False)


sns.histplot(x="log_fare", hue="Embarked", data=training_data, log_scale = False)




