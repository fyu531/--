import math
from collections import defaultdict
from backend.utils import euclidean_distance, manhattan_distance, majority_vote

class KNN:
    """K最近邻算法（支持分类和回归）"""
    def __init__(self, k=5, distance_metric='euclidean', task_type='classification'):
        self.k = k
        self.distance_metric = distance_metric
        self.task_type = task_type
        self.X_train = None
        self.y_train = None
    
    def fit(self, features, labels):
        """训练KNN模型（存储数据）"""
        if len(features) != len(labels):
            raise ValueError("特征和标签的数量必须相同")
        
        if len(features) == 0:
            raise ValueError("数据集不能为空")
        
        self.X_train = features
        self.y_train = labels
    
    # 新增train方法，兼容统一接口
    def train(self, features, labels):
        """为兼容统一接口，调用fit方法"""
        self.fit(features, labels)
    
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
            raise ValueError("KNN模型尚未训练，请先调用fit方法")
        
        distances = []
        for x, y in zip(self.X_train, self.y_train):
            dist = self._calculate_distance(sample, x)
            distances.append((dist, y))
        
        distances.sort()
        neighbors = distances[:self.k]
        neighbor_labels = [label for (dist, label) in neighbors]
        
        if self.task_type == 'classification':
            return majority_vote(neighbor_labels)
        else:  # regression
            return sum(neighbor_labels) / len(neighbor_labels)
    
    def predict(self, features):
        """预测多个样本"""
        return [self._predict_sample(sample) for sample in features]
    
    def get_visualization_data(self):
        """获取KNN可视化数据"""
        if self.X_train is None:
            return None
            
        return {
            'k': self.k,
            'distance_metric': self.distance_metric,
            'task_type': self.task_type,
            'train_samples_count': len(self.X_train)
        }