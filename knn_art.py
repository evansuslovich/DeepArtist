import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# fetching the data from csv files
art_train = pd.read_csv("art_train.csv")
art_test = pd.read_csv("art_test.csv")

# remove index and filename (28.5%)
art_train = art_train.drop(columns=art_train.columns[0]).drop(columns=art_train.columns[1])
art_test = art_test.drop(columns=art_test.columns[0]).drop(columns=art_test.columns[1])

# color palette only (There are 5 colors in the palette and each RGB value) (22.25%)
art_train_color = art_train.drop(
    columns=["Channel 1 Mean", "Channel 1 Standard Deviation", "Channel 1 Skewness", "Channel 2 Mean",
             "Channel 2 Standard Deviation", "Channel 2 Skewness", "Channel 3 Mean", "Channel 3 Standard Deviation",
             "Channel 3 Skewness", "Color Richness", "0 Degrees", "45 Degrees", "90 Degrees", "135 Degrees",
             "180 Degrees", "225 Degrees", "270 Degrees", "315 Degrees"])
art_test_color = art_test.drop(
    columns=["Channel 1 Mean", "Channel 1 Standard Deviation", "Channel 1 Skewness", "Channel 2 Mean",
             "Channel 2 Standard Deviation", "Channel 2 Skewness", "Channel 3 Mean", "Channel 3 Standard Deviation",
             "Channel 3 Skewness", "Color Richness", "0 Degrees", "45 Degrees", "90 Degrees", "135 Degrees",
             "180 Degrees", "225 Degrees", "270 Degrees", "315 Degrees"])

# GLCM (Gray-Level Co-Occurrence Matrix) and Color Palette (23.25%)
art_train_GLCM_colors = art_train.drop(
    columns=["Channel 1 Mean", "Channel 1 Standard Deviation", "Channel 1 Skewness", "Channel 2 Mean",
             "Channel 2 Standard Deviation", "Channel 2 Skewness", "Channel 3 Mean", "Channel 3 Standard Deviation",
             "Channel 3 Skewness", "Color Richness"])
art_test_GLCM_colors = art_test.drop(
    columns=["Channel 1 Mean", "Channel 1 Standard Deviation", "Channel 1 Skewness", "Channel 2 Mean",
             "Channel 2 Standard Deviation", "Channel 2 Skewness", "Channel 3 Mean", "Channel 3 Standard Deviation",
             "Channel 3 Skewness", "Color Richness"])


# GLCM (Gray-Level Co-Occurrence Matrix) only (23%)
art_train_GLCM_only = art_train.drop(columns=["Channel 1 Mean","Channel 1 Standard Deviation","Channel 1 Skewness","Channel 2 Mean","Channel 2 Standard Deviation","Channel 2 Skewness","Channel 3 Mean","Channel 3 Standard Deviation","Channel 3 Skewness","Color Richness", "Color Palette 1 Red","Color Palette 1 Green","Color Palette 1 Blue","Color Palette 2 Red","Color Palette 2 Green","Color Palette 2 Blue","Color Palette 3 Red","Color Palette 3 Green","Color Palette 3 Blue","Color Palette 4 Red","Color Palette 4 Green","Color Palette 4 Blue","Color Palette 5 Red","Color Palette 5 Green","Color Palette 5 Blue"])
art_test_GLCM_only = art_test.drop(columns=["Channel 1 Mean","Channel 1 Standard Deviation","Channel 1 Skewness","Channel 2 Mean","Channel 2 Standard Deviation","Channel 2 Skewness","Channel 3 Mean","Channel 3 Standard Deviation","Channel 3 Skewness","Color Richness", "Color Palette 1 Red","Color Palette 1 Green","Color Palette 1 Blue","Color Palette 2 Red","Color Palette 2 Green","Color Palette 2 Blue","Color Palette 3 Red","Color Palette 3 Green","Color Palette 3 Blue","Color Palette 4 Red","Color Palette 4 Green","Color Palette 4 Blue","Color Palette 5 Red","Color Palette 5 Green","Color Palette 5 Blue"])

# color moments only (26.5%)
art_train_color_moments_only = art_train.drop(columns=["Color Palette 1 Red","Color Palette 1 Green","Color Palette 1 Blue","Color Palette 2 Red","Color Palette 2 Green","Color Palette 2 Blue","Color Palette 3 Red","Color Palette 3 Green","Color Palette 3 Blue","Color Palette 4 Red","Color Palette 4 Green","Color Palette 4 Blue","Color Palette 5 Red","Color Palette 5 Green","Color Palette 5 Blue", "0 Degrees","45 Degrees","90 Degrees","135 Degrees","180 Degrees","225 Degrees","270 Degrees","315 Degrees"])
art_test_color_moments_only = art_test.drop(columns=["Color Palette 1 Red","Color Palette 1 Green","Color Palette 1 Blue","Color Palette 2 Red","Color Palette 2 Green","Color Palette 2 Blue","Color Palette 3 Red","Color Palette 3 Green","Color Palette 3 Blue","Color Palette 4 Red","Color Palette 4 Green","Color Palette 4 Blue","Color Palette 5 Red","Color Palette 5 Green","Color Palette 5 Blue", "0 Degrees","45 Degrees","90 Degrees","135 Degrees","180 Degrees","225 Degrees","270 Degrees","315 Degrees"])


# color richness only (14.5%)
columns_to_select = ['Color Richness', 'Genre']
art_train_color_richness_only = art_train[columns_to_select].copy()
art_test_color_richness_only = art_test[columns_to_select].copy()


# Load training data and testing data into pandas DataFrames
train_data = art_train_GLCM_colors
test_data = art_test_GLCM_colors

# Split training data into features (X_train) and labels (y_train)
X_train = train_data.drop(columns=["Genre"])
y_train = train_data["Genre"]

# # Split testing data into features (X_test) and labels (y_test)
X_test = test_data.drop(columns=["Genre"])
y_test = test_data["Genre"]
# Standardize the feature values using the same scaler for both sets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a KNN classifier
scores = []
for k in range(1, 51):
    print(k)
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train_scaled, y_train)
    y_pred = knn_classifier.predict(X_test_scaled)
    scores.append(accuracy_score(y_test, y_pred))


plt.figure(figsize=(10, 6))
plt.plot(range(1, 51), scores, marker='o')
plt.title('KNN Classifier Accuracy vs. Number of Neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy Score')
plt.xticks(range(1, 51))
plt.grid(True)


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=knn_classifier.classes_, yticklabels=knn_classifier.classes_)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

