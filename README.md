# Credit-Card-Fraud-Detection

##  Steps to complete the project, including splitting the data into training and testing sets, training a logistic regression model, and evaluating its performance.

#### Importing the Dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


Loading and Viewing the Dataset
# loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv('/content/credit_data.csv')

# first 5 rows of the dataset
print(credit_card_data.head())

# last 5 rows of the dataset
print(credit_card_data.tail())

# dataset information
print(credit_card_data.info())

# checking the number of missing values in each column
print(credit_card_data.isnull().sum())

# distribution of legit transactions & fraudulent transactions
print(credit_card_data['Class'].value_counts())



