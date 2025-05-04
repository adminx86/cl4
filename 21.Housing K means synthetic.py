'''
 To identify regional housing market segments, an analyst wants to group properties 
based on similar features. Apply the K-Means clustering algorithm to segment the 
housing dataset and visualize the clusters.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns

# Step 1: Generate synthetic housing data
np.random.seed(42)
n_samples = 300

area = np.random.randint(800, 4000, size=n_samples)
bedrooms = np.random.randint(1, 6, size=n_samples)
age = np.random.randint(0, 30, size=n_samples)
price = area * np.random.uniform(100, 300) - age * 1000 + bedrooms * 5000 + np.random.normal(0, 10000, n_samples)

df = pd.DataFrame({
    'area': area,
    'bedrooms': bedrooms,
    'age': age,
    'price': price
})

# Step 2: Preprocessing
features = ['area', 'bedrooms', 'age', 'price']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Step 4: Reduce dimensions for visualization using PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)
df['PC1'] = pca_components[:, 0]
df['PC2'] = pca_components[:, 1]

# Step 5: Plot clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='cluster', palette='Set1', s=100)
plt.title('K-Means Clustering of Houses')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()
