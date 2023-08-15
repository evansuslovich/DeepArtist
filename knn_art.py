import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
scores = []
for k in range(1, 25):
    print(k)
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train_scaled, y_train)
    y_pred = knn_classifier.predict(X_test_scaled)
    scores.append(accuracy_score(y_test, y_pred))



# param_grid = {
#     'n_neighbors': range(1, 50),
#     'weights': ['uniform', 'distance']
# }
#
# grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10, scoring='accuracy')
# grid_search.fit(X_train_scaled, y_train)
#
# best_params = grid_search.best_params_
# best_accuracy = grid_search.best_score_
#
# print("Best Parameters:", best_params)
# print("Best Accuracy:", best_accuracy)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 25), scores, marker='o')
plt.title('KNN Classifier Accuracy vs. Number of Neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy Score')
plt.xticks(range(1, 25))
plt.grid(True)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=knn_classifier.classes_, yticklabels=knn_classifier.classes_)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

