'''
Perform the data clustering algorithm using any Clustering algorithm
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -----------------------------
# Step 1: Load the Dataset
# -----------------------------
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# -----------------------------
# Step 2: Apply K-Means Clustering
# -----------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
clusters = kmeans.labels_

# Add the cluster labels to the dataframe
X['Cluster'] = clusters

# -----------------------------
# Step 3: Dimensionality Reduction for Visualization
# -----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.iloc[:, :-1])
X['PCA1'] = X_pca[:, 0]
X['PCA2'] = X_pca[:, 1]

# -----------------------------
# Step 4: Visualize the Clusters
# -----------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(data=X, x='PCA1', y='PCA2', hue='Cluster', palette='Set1', s=100)
plt.title('K-Means Clustering on Iris Dataset')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()