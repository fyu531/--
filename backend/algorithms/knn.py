import math
from collections import defaultdict
from backend.utils import euclidean_distance, manhattan_distance, majority_vote

class KNN:
    """K最近邻分类器"""
    def __init__(self, n_neighbors=5, distance_metric='euclidean'):
        # 近邻数量
        self.n_neighbors = n_neighbors
        # 距离度量 ('euclidean' 或 'manhattan')
        self.distance_metric = distance_metric
        # 训练数据
        self.X_train = None
        self.y_train = None
    
    def fit(self, features, labels):
        """训练KNN分类器（实际上只是存储数据）"""
        if len(features) != len(labels):
            raise ValueError("特征和标签的数量必须相同")
        
        if len(features) == 0:
            raise ValueError("数据集不能为空")
        
        self.X_train = features
        self.y_train = labels
    
    def _calculate_distance(self, x1, x2):
        """计算两个样本之间的距离"""
        if self.distance_metric == 'euclidean':
            return euclidean_distance(x1, x2)
        elif self.distance_metric == 'manhattan':
            return manhattan_distance(x1, x2)
        else:
            raise ValueError(f"不支持的距离度量: {self.distance_metric}")
    
    def _predict_sample(self, sample):
        """预测单个样本"""
        if self.X_train is None or self.y_train is None:
            raise ValueError("KNN分类器尚未训练，请先调用fit方法")
        
        # 计算与所有训练样本的距离
        distances = []
        for x, y in zip(self.X_train, self.y_train):
            dist = self._calculate_distance(sample, x)
            distances.append((dist, y))
        
        # 按距离排序并选择最近的k个样本
        distances.sort()
        neighbors = distances[:self.n_neighbors]
        neighbor_labels = [label for (dist, label) in neighbors]
        
        # 多数投票决定预测结果
        return majority_vote(neighbor_labels)
    
    def predict(self, features):
        """预测多个样本"""
        return [self._predict_sample(sample) for sample in features]
