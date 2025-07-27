import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print ('Accuracy: ', accuracy*100)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Print the full confusion matrix
print('\nConfusion Matrix:')
print(cm)

# Print individual elements of confusion matrix
print('\nIndividual Elements:')
print(f'True Positives (Class 0): {cm[0,0]}')
print(f'True Positives (Class 1): {cm[1,1]}')
print(f'True Positives (Class 2): {cm[2,2]}')

# new code for print branch
print ('this is a print statement for print branch')