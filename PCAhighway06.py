import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
data = pd.read_csv(r"C:\Users\Malak Amireh\Downloads\ground_truth (1).csv")

# Selecting features for PCA - excluding non-relevant columns
features = data.columns.drop(['time', 'state'])  # Adjust based on your dataset's specific columns

# Standardizing the features
x = data.loc[:, features].values
x = StandardScaler().fit_transform(x)

# Performing PCA
pca = PCA(n_components=None)  # None means all components are kept
pca.fit(x)

# Calculating the explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = explained_variance.cumsum()

# Transforming the data using the first 10 principal components
pca_10 = PCA(n_components=10)  # Keeping first 10 components
transformed_data = pca_10.fit_transform(x)

# Scree Plot
plt.figure(figsize=(10, 6))
plt.bar(range(1, 11), explained_variance[:10], alpha=0.6, color='b', label='Individual explained variance')
plt.step(range(1, 11), cumulative_explained_variance[:10], where='mid', label='Cumulative explained variance')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.title('Scree Plot')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Cumulative Explained Variance Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), cumulative_explained_variance[:10], marker='o', linestyle='-', color='r')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.tight_layout()
plt.show()

# Biplot
fig, ax = plt.subplots(figsize=(10, 6))
pc1 = 0
pc2 = 1
ax.scatter(transformed_data[:, pc1], transformed_data[:, pc2], alpha=0.5, color='g')
for i, feature in enumerate(features):
    ax.arrow(0, 0, pca_10.components_[pc1, i]*max(transformed_data[:, pc1]), pca_10.components_[pc2, i]*max(transformed_data[:, pc2]), head_width=0.02, head_length=0.03, fc='r', ec='r')
    ax.text(pca_10.components_[pc1, i]*max(transformed_data[:, pc1])*1.2, pca_10.components_[pc2, i]*max(transformed_data[:, pc2])*1.2, feature, color='b')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Biplot')
plt.grid(True)
plt.tight_layout()
plt.show()

# 3D Scatter Plot of Transformed Data
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2], c='purple', marker='o', alpha=0.5)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('3D Scatter Plot of PCA Transformed Data')
plt.tight_layout()
plt.show()
