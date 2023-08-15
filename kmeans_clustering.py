import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data into a pandas DataFrame
data = pd.read_csv("art_train.csv")

# Drop non-numeric columns
numeric_data = data.drop(columns=["Filename", "Genre"])

# Standardize the feature values
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Apply KMeans clustering
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve to determine the optimal number of clusters
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Curve for KMeans Clustering')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# Based on the elbow curve, choose an appropriate number of clusters
chosen_k = ...

# Apply KMeans clustering with the chosen number of clusters
kmeans = KMeans(n_clusters=chosen_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Now you can analyze the clusters and their characteristics
# For example, you can group by the 'Cluster' column and compute cluster statistics
cluster_stats = data.groupby('Cluster').mean()

# Visualize the results as needed
# For example, you could create scatter plots based on two features and color by cluster
plt.figure(figsize=(10, 6))
plt.scatter(data['Feature1'], data['Feature2'], c=data['Cluster'], cmap='rainbow')
plt.title('KMeans Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()
