import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load training data and testing data into pandas DataFrames
train_data = pd.read_csv("art_train.csv")
test_data = pd.read_csv("art_test.csv")

# Split training data into features (X_train) and labels (y_train)
X_train = train_data.drop(columns=["Filename"])
y_train = train_data["Genre"]

# Split testing data into features (X_test) and labels (y_test)
X_test = test_data.drop(columns=["Filename"])
y_test = test_data["Genre"]

# Standardize the feature values using the same scaler for both sets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a KNN classifier
k = 50  # Number of neighbors to consider
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Train the classifier on the training data
knn_classifier.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = knn_classifier.predict(X_test_scaled)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
