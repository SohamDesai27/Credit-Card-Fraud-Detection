# Credit-Card-Fraud-Detection

Dataset: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](url)

Steps to complete the project, including splitting the data into training and testing sets, training a logistic regression model, and evaluating its performance.

### Importing the Dependencies

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

### Loading and Viewing the Dataset

```python
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
```

### Separating the Data for Analysis

```python
# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print(legit.shape)
print(fraud.shape)

# statistical measures of the data
print(legit.Amount.describe())
print(fraud.Amount.describe())

# compare the values for both transactions
print(credit_card_data.groupby('Class').mean())
```

### Under-Sampling

```python
# Build a sample dataset containing similar distribution of normal transactions and Fraudulent Transactions
# Number of Fraudulent Transactions --> 492
legit_sample = legit.sample(n=492)

# Concatenating two DataFrames
new_dataset = pd.concat([legit_sample, fraud], axis=0)

print(new_dataset.head())
print(new_dataset.tail())
```

### Splitting the Data into Features and Target

```python
# Splitting the data into Features and Target
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

print(X.head())
print(Y.head())
```

### Splitting the Data into Training and Testing Sets

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X_train.shape)
print(X_test.shape)
```

### Training the Model

```python
# Model Training with Logistic Regression
model = LogisticRegression()

# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)
```

### Model Evaluation

```python
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data: ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data: ', test_data_accuracy)
```

### Complete Script

Here is the complete script that combines all the steps:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv('/content/credit_data.csv')

# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

# Build a sample dataset containing similar distribution of normal transactions and Fraudulent Transactions
legit_sample = legit.sample(n=492)

# Concatenating two DataFrames
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# Splitting the data into Features and Target
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

# Splitting the data into Training and Testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Model Training with Logistic Regression
model = LogisticRegression()
model.fit(X_train, Y_train)

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data: ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data: ', test_data_accuracy)
```
