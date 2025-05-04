'''
 To identify regional housing market segments, an analyst wants to group properties 
based on similar features. Apply the K-Means clustering algorithm to segment the 
housing dataset and visualize the clusters.
'''
# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# Replace 'housing_data.csv' with the path to your dataset
df = pd.read_csv('housing_data.csv')

# Inspect the first few rows of the dataset
print(df.head())

# Step 1: Preprocess the data
# Here, we'll assume we want to use numerical features for clustering.
# Select relevant features (modify according to your dataset)
# You can replace 'feature_1', 'feature_2', etc., with the actual column names of your dataset
features = df[['feature_1', 'feature_2', 'feature_3', 'feature_4']]  # Replace with actual columns

# Step 2: Normalize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 3: Apply K-Means Clustering
# First, we determine the number of clusters (k) using the Elbow method
wcss = []  # Within-cluster sum of squares
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow graph to identify the optimal number of clusters
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()

# Step 4: Train the K-Means model with the optimal number of clusters (say, k=3)
# You can adjust the number of clusters based on the elbow method result
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(scaled_features)

# Step 5: Visualize the clusters (2D visualization for simplicity)
# If you have more than two features, consider using PCA or t-SNE for dimensionality reduction

# Create a new DataFrame with cluster labels
df['Cluster'] = y_kmeans

# Visualize the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['feature_1'], y=df['feature_2'], hue=df['Cluster'], palette='viridis', s=100, alpha=0.7)
plt.title('K-Means Clusters')
plt.xlabel('Feature 1')  # Replace with actual feature name
plt.ylabel('Feature 2')  # Replace with actual feature name
plt.legend(title='Cluster')
plt.show()

# Optionally, you can save the DataFrame with cluster labels to a new CSV file
df.to_csv('housing_with_clusters.csv', index=False)
