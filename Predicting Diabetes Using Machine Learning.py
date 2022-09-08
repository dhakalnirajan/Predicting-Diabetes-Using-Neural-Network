# Predicting Diabetes of a Patient Using Machine Learning

""""
Diabetes is a disease that occurs when your blood glucose,
also called the blood sugar, is too high in the body. Blood
glucose is a main source of energy generating in the body to
function. To maintain the level of blood sugar, the pancreas
in the digestive system secrets an hormone called Insulin.
Insulin helps blood sugar enter the body's cells so it can be
used for energy. Insulin also signals the liver to store blood
sugar for later use. Blood sugar enters cells, and levels in
the bloodstream decrease, signaling insulin to decrease too.
When the level of Insulin is not produced sufficiently to
neutralize blood sugar by some dysfunction, the sugar level
increases with time and the disease develops.
"""

# Importing the Dependencies

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Collecting and Pre-Processing the Data

#loading the dataset to a pandas Dataframe
diabetes_dataset = pd.read_csv('diabetes_data.csv')

pd.read_csv?

#printing the first five rows of dataset 
diabetes_dataset.head()

#printing the last five rows of dataset 
diabetes_dataset.tail()

#numbers of rows and columns in this dataset
diabetes_dataset.shape

#getting the statistical measures of the data 
diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()

"""
0 ==== > Non-Diabetes

1 ==== > Diabetes
"""

diabetes_dataset.groupby('Outcome').mean()

print(X)
print(Y)

# Standardizing the Data
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
print(standardized_data)

X = standardized_data
Y = diabetes_dataset['Outcome']

print(X)
print(Y)

# Spliting the Training and Test

X_train, X_test, Y_train,Y_test = train_test_split(X,Y, test_size=0.2, stratify = Y, random_state=2)

print(X.shape, X_train.shape,X_test.shape)

# Training The Model

classifier = svm.SVC(kernel='linear')

#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

# Evaluating Model
#Checking the Accuracy Score

#accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print("Accuracy score of the training data is", training_data_accuracy)