import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

np.random.seed(9)  # 设置随机种子以确保每次生成相同的随机数据

# 生成 A 类、B 类、C 类数据点
# A 类数据以 (0, 0) 为中心，协方差矩阵为 [[1, 0], [0, 0.5]]
A = np.random.multivariate_normal([0, 0], [[1, 0], [0, 0.5]], 25)

# B 类数据以 (3, 0) 为中心，协方差矩阵为 [[0.5, 0], [0, 1]]
B = np.random.multivariate_normal([3, 0], [[0.5, 0], [0, 1]], 25)

# C 类数据以 (0, 2) 为中心，协方差矩阵为 [[1, 0.3], [0.3, 1]]
C = np.random.multivariate_normal([0, 2], [[1, 0.3], [0.3, 1]], 25)

# 合并 A、B、C 类数据到一个大的数据集中
data = np.vstack((A, B, C))

# 设置每个类的真实标签（A类为0，B类为1，C类为2）
ground_truth = np.array([0]*25 + [1]*25 + [2]*25)

# 自定义的欧几里得距离计算函数，计算每个数据点与质心的距离
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

# K-Means 聚类算法实现，包含三种初始化方式（随机、最远点、K-Means++）
def kmeans(X, k, init='random', max_iters=100):
    if init == 'random':  # 随机初始化
        # 从数据集中随机选择 k 个点作为初始质心
        centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    elif init == 'farthest':  # 最远点初始化
        # 随机选择一个点作为第一个质心
        centroids = [X[np.random.choice(X.shape[0])]]
        for _ in range(1, k):  # 逐步选择其他质心
            # 计算每个点到最近的已有质心的距离
            dist = np.min([euclidean_distance(X, np.array(c)) for c in centroids], axis=0)
            # 选择距离最远的点作为新的质心
            next_centroid = X[np.argmax(dist)]
            centroids.append(next_centroid)
        centroids = np.array(centroids)
    elif init == 'kmeans++':  # K-Means++ 初始化
        # 随机选择第一个质心
        centroids = [X[np.random.choice(X.shape[0])]]
        for _ in range(1, k):
            # 计算每个点到最近质心的平方距离
            dist_sq = np.min([euclidean_distance(X, np.array(c)) ** 2 for c in centroids], axis=0)
            # 按距离平方的比例选择下一个质心
            probs = dist_sq / np.sum(dist_sq)
            # 生成累计概率
            cumulative_probs = np.cumsum(probs)
            # 使用随机数选择新质心
            r = np.random.rand()
            next_centroid = X[np.where(cumulative_probs >= r)[0][0]]
            centroids.append(next_centroid)
        centroids = np.array(centroids)

    # 迭代优化质心
    for _ in range(max_iters):
        # 计算每个点到质心的距离
        distances = np.array([euclidean_distance(X, c) for c in centroids])
        # 将每个点分配到最近的质心所属的簇中
        clusters = np.argmin(distances, axis=0)
        # 重新计算每个簇的新质心
        new_centroids = np.array([X[clusters == j].mean(axis=0) for j in range(k)])
        # 检查质心是否已经收敛（即新质心和旧质心没有明显变化）
        if np.allclose(new_centroids, centroids):
            break
        # 更新质心以继续迭代
        centroids = new_centroids
        
    return clusters, centroids

# 手动匹配聚类结果的标签与真实标签
def match_labels(true_labels, pred_labels):
    best_accuracy = 0  # 初始化最佳准确率
    best_matching = pred_labels.copy()  # 保存最优匹配的标签结果
    # 遍历标签 [0, 1, 2] 的所有排列组合
    for perm in permutations([0, 1, 2]):
        # 根据当前排列生成新的预测标签
        perm_labels = np.array([perm[label] for label in pred_labels])
        # 计算当前排列的准确率
        accuracy = np.mean(perm_labels == true_labels)
        # 如果当前排列的准确率更高，则更新最优结果
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_matching = perm_labels
    return best_matching

# 计算聚类的准确率
def calculate_accuracy(true_labels, pred_labels):
    # 通过标签匹配找到最佳的标签对齐方式
    matched_labels = match_labels(true_labels, pred_labels)
    # 计算最终准确率
    accuracy = np.mean(matched_labels == true_labels)
    return accuracy

# 使用三种不同的初始化方法执行 K-Means 聚类
clusters_random, centroids_random = kmeans(data, 3, init='random')
clusters_farthest, centroids_farthest = kmeans(data, 3, init='farthest')
clusters_kmeanspp, centroids_kmeanspp = kmeans(data, 3, init='kmeans++')

# 绘制初始质心和聚类结果
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# 绘制真实簇分布
axes[0].scatter(data[:, 0], data[:, 1], c=ground_truth, cmap='rainbow', s=50)
axes[0].set_title('True Clusters')

# 分别绘制三种初始化方法的聚类结果
for ax, clusters, centroids, title in zip(
    axes[1:], 
    [clusters_random, clusters_farthest, clusters_kmeanspp], 
    [centroids_random, centroids_farthest, centroids_kmeanspp], 
    ['Random Initialization', 'Farthest Point Initialization', 'K-Means++ Initialization']
):
    # 绘制数据点，并根据簇颜色区分
    ax.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis', s=50)
    # 绘制质心
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)
    ax.set_title(title)

plt.tight_layout()
plt.show()

# 计算每种初始化方法的准确率
accuracy_random = calculate_accuracy(ground_truth, clusters_random)
accuracy_farthest = calculate_accuracy(ground_truth, clusters_farthest)
accuracy_kmeanspp = calculate_accuracy(ground_truth, clusters_kmeanspp)

# 打印每种初始化方法的准确率
print(f"Accuracy (Random Initialization): {accuracy_random:.2f}")
print(f"Accuracy (Farthest Point Initialization): {accuracy_farthest:.2f}")
print(f"Accuracy (K-Means++ Initialization): {accuracy_kmeanspp:.2f}")