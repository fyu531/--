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
        n_samples, n_features = X.shape
        centroids = np.zeros((self.k, n_features))
        empty_clusters = []
        
        for i, cluster in enumerate(clusters):
            if cluster:
                centroids[i] = np.mean(X[cluster], axis=0)
            else:
                empty_clusters.append(i)
                
        # 处理空聚类：从所有样本中重新随机选择中心
        if empty_clusters:
            used_indices = set()
            for cluster in clusters:
                used_indices.update(cluster)
                
            unused_indices = list(set(range(n_samples)) - used_indices)
            if unused_indices:
                # 从未被选中的样本中随机选择
                new_centroid_indices = random.sample(unused_indices, len(empty_clusters))
                for i, idx in zip(empty_clusters, new_centroid_indices):
                    centroids[i] = X[idx]
            else:
                # 所有样本都已被使用，从整个数据集随机选择
                new_centroid_indices = random.sample(range(n_samples), len(empty_clusters))
                for i, idx in zip(empty_clusters, new_centroid_indices):
                    centroids[i] = X[idx]
                    
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
        if self.centroids is None or self.clusters is None:
            return None
        
        # 计算总样本数量（从聚类结果反推）
        total_samples = sum(len(cluster) for cluster in self.clusters)
        
        # 创建与样本数量匹配的标签数组
        labels = np.zeros(total_samples, dtype=int)
        
        # 为每个样本分配聚类标签
        for cluster_idx, sample_indices in enumerate(self.clusters):
            for sample_idx in sample_indices:
                # 确保样本索引在有效范围内
                if sample_idx < total_samples:
                    labels[sample_idx] = cluster_idx
                
        return {
            'k': self.k,
            'centroids': self.centroids.tolist(),
            'labels': labels.tolist(),  # 每个样本的聚类标签
            'cluster_sizes': [len(cluster) for cluster in self.clusters]
        }
