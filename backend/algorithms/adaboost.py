import numpy as np
from .decision_tree import DecisionTree  # 使用决策树作为弱分类器

class AdaBoost:
    """AdaBoost算法实现（分类）"""
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.estimators = []  # 存储弱分类器
        self.estimator_weights = []  # 存储弱分类器权重
        
    def train(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        n_samples = X.shape[0]
        
        # 将标签转换为-1和1
        y_transformed = np.where(y == 0, -1, 1)
        
        # 初始化样本权重（确保是numpy数组）
        sample_weights = np.full(n_samples, (1 / n_samples), dtype=np.float64)
        
        self.estimators = []
        self.estimator_weights = []
        
        for _ in range(self.n_estimators):
            estimator = DecisionTree(max_depth=1)
            estimator.train(X, y_transformed, sample_weights)
            
            # 预测并确保结果是numpy数组
            y_pred = np.array(estimator.predict(X), dtype=np.float64)
            
            # 计算错误率（确保所有操作数都是numpy数组）
            incorrect = (y_pred != y_transformed).astype(np.float64)
            error = np.sum(sample_weights * incorrect) / np.sum(sample_weights)
            
            # 计算分类器权重
            estimator_weight = 0.5 * np.log((1 - error) / (error + 1e-10))
            
            # 更新样本权重
            sample_weights *= np.exp(-estimator_weight * y_transformed * y_pred)
            sample_weights = np.maximum(sample_weights, 1e-10)  # 确保非负
            sample_weights /= np.sum(sample_weights)  # 归一化
            
            self.estimators.append(estimator)
            self.estimator_weights.append(estimator_weight)
            
    def predict(self, X):
        if not self.estimators:
            raise RuntimeError("模型尚未训练，请先调用train方法")
            
        X = np.array(X, dtype=np.float64)
        final_pred = np.zeros(X.shape[0], dtype=np.float64)
        
        # 累加所有弱分类器的加权预测
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            # 确保预测结果是numpy数组
            y_pred = np.array(estimator.predict(X), dtype=np.float64)
            final_pred += weight * y_pred  # 现在两边都是numpy类型
            
        # 将结果转换为0和1
        return np.where(np.sign(final_pred) == 1, 1, 0)
        
    def get_visualization_data(self):
        if not self.estimators:
            return None
            
        return {
            'n_estimators': self.n_estimators,
            'estimator_weights': self.estimator_weights,
            'estimator_count': len(self.estimators)
        }


