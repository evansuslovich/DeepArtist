import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.metrics import confusion_matrix

art_train = pd.read_csv("art_train.csv")
art_test = pd.read_csv("art_test.csv")

art_train = art_train.drop(columns=art_train.columns[0])
art_test = art_test.drop(columns=art_test.columns[0])

#print(art_test.isnull().sum())


# Load training data and testing data into pandas DataFrames
train_data = art_train
test_data = art_test

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

