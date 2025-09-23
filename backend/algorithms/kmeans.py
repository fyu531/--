import numpy as np
import random

class KMeans:
    """K均值聚类算法实现"""
    def __init__(self, k=2, max_iters=100):
        """
        初始化KMeans模型
        :param k: 聚类数量
        :param max_iters: 最大迭代次数
        """
        self.k = k
        self.max_iters = max_iters
        self.centroids = None  # 聚类中心
        self.clusters = None   # 聚类结果
        
    def _initialize_centroids(self, X):
        """初始化聚类中心（随机选择k个样本）"""
        n_samples, n_features = X.shape
        centroids = np.zeros((self.k, n_features))
        
        # 随机选择k个不同的样本作为初始中心
        indices = random.sample(range(n_samples), self.k)
        for i, idx in enumerate(indices):
            centroids[i] = X[idx]
            
        return centroids
        
    def _assign_clusters(self, X, centroids):
        """将样本分配到最近的聚类中心"""
        clusters = [[] for _ in range(self.k)]
        
        for idx, sample in enumerate(X):
            # 计算到每个中心的距离
            distances = [np.linalg.norm(sample - centroid) for centroid in centroids]
            # 找到最近的中心
            closest_centroid_idx = np.argmin(distances)
            clusters[closest_centroid_idx].append(idx)
            
        return clusters
        
    def _update_centroids(self, X, clusters):
        """根据聚类结果更新聚类中心"""
        n_features = X.shape[1]
        centroids = np.zeros((self.k, n_features))
        
        for i, cluster in enumerate(clusters):
            if cluster:  # 避免空聚类
                cluster_samples = X[cluster]
                centroids[i] = np.mean(cluster_samples, axis=0)
                
        return centroids
        
    def _is_converged(self, old_centroids, new_centroids, tolerance=1e-4):
        """检查是否收敛（聚类中心变化小于阈值）"""
        distances = [np.linalg.norm(old_centroids[i] - new_centroids[i]) for i in range(self.k)]
        return sum(distances) < tolerance
        
    def train(self, X):
        """
        训练KMeans模型
        :param X: 特征数据（无标签）
        """
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # 初始化聚类中心
        self.centroids = self._initialize_centroids(X)
        
        # 迭代更新
        for _ in range(self.max_iters):
            # 分配样本到聚类
            self.clusters = self._assign_clusters(X, self.centroids)
            
            # 保存当前中心
            old_centroids = self.centroids.copy()
            
            # 更新聚类中心
            self.centroids = self._update_centroids(X, self.clusters)
            
            # 检查是否收敛
            if self._is_converged(old_centroids, self.centroids):
                break
                
    def predict(self, X):
        """
        预测样本所属聚类
        :param X: 样本数据
        :return: 聚类索引
        """
        if self.centroids is None:
            raise RuntimeError("模型尚未训练，请先调用train方法")
            
        X = np.array(X)
        predictions = []
        
        for sample in X:
            distances = [np.linalg.norm(sample - centroid) for centroid in self.centroids]
            closest_centroid_idx = np.argmin(distances)
            predictions.append(closest_centroid_idx)
            
        return np.array(predictions)
        
    def get_visualization_data(self):
        """获取KMeans可视化数据"""
        if self.centroids is None:
            return None
            
        return {
            'k': self.k,
            'centroids': self.centroids.tolist(),
            'cluster_sizes': [len(cluster) for cluster in self.clusters] if self.clusters else None
        }
