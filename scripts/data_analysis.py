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
import os

# Initiating contants for paths
path = os.getcwd()
dataset_path = os.path.join(*os.path.split(path)[:-1],"datasets")
training_data = os.path.join(dataset_path,"train.csv")


# Importing the training data
training_data = pd.read_csv(training_data)

training_data = training_data.astype({"Pclass":str})

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
sns.catplot(x="Pclass", col="Survived", data=training_data, ci = None, kind = "count",order = ["1","2","3"])
plt.show()

# The sex of the passenger
sns.catplot(x="Sex", col="Survived", data=training_data, ci = None, kind = "count")
plt.show()

# The place where the passenger embarked
sns.catplot(x="Embarked", col="Survived", data=training_data, ci = None, kind = "count", order = ["S","C","Q"])
plt.show()

# Plotting the same categories but this time, it is the percentage of how many survived in each category
    # This is done in order to
sns.barplot(x="Pclass", y="Survived", data=training_data, ci = None, order = ["1","2","3"])
plt.show()

sns.barplot(x="Sex", y="Survived", data=training_data, ci = None)
plt.show()

sns.barplot(x="Embarked", y="Survived", data=training_data, ci = None)
plt.show()

# Here the survival rate plot for the titles are added as well
sns.barplot(x="title", y="Survived", data=training_data, ci = None)
plt.xticks(rotation=90)
plt.show()

sns.histplot(x="Age", hue="Survived", data=training_data)
plt.show()

sns.histplot(x="SibSp", hue="Survived", data=training_data, bins = 8)
plt.show()


sns.histplot(x="Parch", hue="Survived", data=training_data, bins = 6)
plt.show()

sns.histplot(x="Fare", hue="Survived", data=training_data, log_scale = False)
plt.show()

sns.histplot(x="log_fare", hue="Survived", data=training_data, log_scale = False)
plt.show()

sns.histplot(x="log_fare", hue="Embarked", data=training_data, log_scale = False)
plt.show()


sns.displot(training_data,x="Embarked",y="Pclass", legend = True, kind = "hist")
plt.show()


general_survival_rate = np.mean(training_data["Survived"].tolist())

male_survival_rate = np.mean(training_data[training_data["Sex"] == "male"]["Survived"].tolist())

female_survival_rate = np.mean(training_data[training_data["Sex"] == "female"]["Survived"].tolist())

male_3rd_class_survival_percentage = np.mean(training_data[(training_data["Sex"] == "male") & (training_data["Pclass"] == "3")]["Survived"].tolist())
male_2nd_class_survival_percentage = np.mean(training_data[(training_data["Sex"] == "male") & (training_data["Pclass"] == "2")]["Survived"].tolist())
male_1st_class_survival_percentage = np.mean(training_data[(training_data["Sex"] == "male") & (training_data["Pclass"] == "1")]["Survived"].tolist())

print(male_1st_class_survival_percentage,male_2nd_class_survival_percentage,male_3rd_class_survival_percentage)

female_1st_class_survival_percentage = np.mean(training_data[(training_data["Sex"] == "female") & (training_data["Pclass"] == "1")]["Survived"].tolist())
female_2nd_class_survival_percentage = np.mean(training_data[(training_data["Sex"] == "female") & (training_data["Pclass"] == "2")]["Survived"].tolist())
female_3rd_class_survival_percentage = np.mean(training_data[(training_data["Sex"] == "female") & (training_data["Pclass"] == "3")]["Survived"].tolist())