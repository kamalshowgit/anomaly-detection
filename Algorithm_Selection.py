import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Create a DataFrame
df = pd.read_csv('anomaly_data.csv')

# Split the data into features (X) and target variable (y)
X = df[['Time', 'Metric']]
y = df['Anomaly']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize classifiers
rf_classifier = RandomForestClassifier(random_state=42)
svm_classifier = SVC(random_state=42)
lr_classifier = LogisticRegression(random_state=42)

# Train the classifiers
rf_classifier.fit(X_train, y_train)
svm_classifier.fit(X_train, y_train)
lr_classifier.fit(X_train, y_train)

# Make predictions
rf_predictions = rf_classifier.predict(X_test)
svm_predictions = svm_classifier.predict(X_test)
lr_predictions = lr_classifier.predict(X_test)

# Evaluate performance
rf_accuracy = accuracy_score(y_test, rf_predictions)
svm_accuracy = accuracy_score(y_test, svm_predictions)
lr_accuracy = accuracy_score(y_test, lr_predictions)

rf_precision = precision_score(y_test, rf_predictions)
svm_precision = precision_score(y_test, svm_predictions)
lr_precision = precision_score(y_test, lr_predictions)

rf_recall = recall_score(y_test, rf_predictions)
svm_recall = recall_score(y_test, svm_predictions)
lr_recall = recall_score(y_test, lr_predictions)

rf_f1 = f1_score(y_test, rf_predictions)
svm_f1 = f1_score(y_test, svm_predictions)
lr_f1 = f1_score(y_test, lr_predictions)

rf_conf_mat = confusion_matrix(y_test, rf_predictions)
svm_conf_mat = confusion_matrix(y_test, svm_predictions)
lr_conf_mat = confusion_matrix(y_test, lr_predictions)

# Display results
print("Random Forest Classifier:")
print(f"Accuracy: {rf_accuracy}")
print(f"Precision: {rf_precision}")
print(f"Recall: {rf_recall}")
print(f"F1 Score: {rf_f1}")
print(f"Confusion Matrix:\n{rf_conf_mat}\n")

print("Support Vector Machine (SVM) Classifier:")
print(f"Accuracy: {svm_accuracy}")
print(f"Precision: {svm_precision}")
print(f"Recall: {svm_recall}")
print(f"F1 Score: {svm_f1}")
print(f"Confusion Matrix:\n{svm_conf_mat}\n")

print("Logistic Regression Classifier:")
print(f"Accuracy: {lr_accuracy}")
print(f"Precision: {lr_precision}")
print(f"Recall: {lr_recall}")
print(f"F1 Score: {lr_f1}")
print(f"Confusion Matrix:\n{lr_conf_mat}\n")
