import numpy as np
import matplotlib.pyplot as plt

# 生成随机数据（你可以替换为自己的数据集）
np.random.seed(42)
n_samples = 300  # 样本数
X = np.concatenate([np.random.normal(0, 1, (n_samples//3, 2)),  # 簇 1       0 是均值，1 是标准差    concatentate 是连接
                    np.random.normal(4, 1.5, (n_samples//3, 2)),  # 簇 2
                    np.random.normal(-3, 1, (n_samples//3, 2))])  # 簇 3
print("数据形状:", X.shape)  # (300, 2)

# k-means 函数
def k_means(X, k, max_iters=100):
    # 步骤 1: 随机初始化 k 个中心
    centroids_idx = np.random.choice(X.shape[0], k, replace=False)    #choice 从数组中随机选取元素   X.shape[0] 是样本数  replace=False 不放回抽样
    centroids = X[centroids_idx]
    
    for iter in range(max_iters):
        # 步骤 2: 计算每个点到所有中心的距离，分配到最近的簇
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  #linalg.norm 计算范数  np.newaxis 增加一个维度
        labels = np.argmin(distances, axis=1)  # 每个点的簇标签
        
        # 步骤 3: 更新每个簇的中心（取均值）
        new_centroids = np.array([X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else centroids[i]
                                  for i in range(k)])  # 避免空簇
        
        # 步骤 4: 检查收敛（如果中心变化很小，停止）
        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < 1e-4):
            print(f"在迭代 {iter+1} 次后收敛")
            break
        
        centroids = new_centroids
    
    # 计算最终 WCSS (损失总和)
    wcss = np.sum([np.sum(np.linalg.norm(X[labels == i] - centroids[i], axis=1)**2) for i in range(k)])
    return labels, centroids, wcss

# 运行算法
k = 3
labels, centroids, wcss = k_means(X, k)
print(f"最终 WCSS: {wcss:.2f}")

# 可视化结果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3)
plt.title('k-means 聚类结果 (NumPy 手动实现)')
plt.show()