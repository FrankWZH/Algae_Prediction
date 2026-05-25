from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from Feature_Filtering.Dimensionality_Reduction import feature
import numpy as np


# n_components=0.95
# 主成分散点图（PCA Biplot）
pca = PCA()
reduced_3d = pca.fit_transform(feature)
print(pca.explained_variance_ratio_)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(reduced_3d[:, 0], reduced_3d[:, 1], reduced_3d[:, 2], alpha=0.5)
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
# plt.title('3D PCA Scatter Plot')
plt.show()


# 累积方差解释率图（Cumulative Variance Plot）
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components = len(cumulative_variance)
plt.figure(figsize=(8, 4))
plt.plot(range(1, n_components+1), np.cumsum(pca.explained_variance_ratio_), 'o-', linewidth=2)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')  # 标记目标阈值
plt.grid()
plt.legend()
plt.show()

