import numpy as np
from .decision_tree import DecisionTree  # 使用决策树作为弱分类器

class AdaBoost:
    """AdaBoost算法实现（分类）"""
    def __init__(self, n_estimators=50):
        """
        初始化AdaBoost模型
        :param n_estimators: 弱分类器数量
        """
        self.n_estimators = n_estimators
        self.estimators = []  # 存储弱分类器
        self.estimator_weights = []  # 存储弱分类器权重
        
    def train(self, X, y):
        """
        训练AdaBoost模型
        :param X: 特征数据
        :param y: 标签数据（0或1）
        """
        X = np.array(X)
        y = np.array(y)
        n_samples = X.shape[0]
        
        # 将标签转换为-1和1
        y_transformed = np.where(y == 0, -1, 1)
        
        # 初始化样本权重
        sample_weights = np.full(n_samples, (1 / n_samples))
        
        self.estimators = []
        self.estimator_weights = []
        
        for _ in range(self.n_estimators):
            # 创建并训练弱分类器（使用深度为1的决策树，即决策 stump）
            estimator = DecisionTree(max_depth=1)
            estimator.train(X, y_transformed, sample_weights)
            
            # 预测
            y_pred = estimator.predict(X)
            
            # 计算错误率
            incorrect = (y_pred != y_transformed)
            error = np.sum(sample_weights * incorrect) / np.sum(sample_weights)
            
            # 计算分类器权重
            estimator_weight = 0.5 * np.log((1 - error) / (error + 1e-10))  # 加小值避免除零
            
            # 更新样本权重
            sample_weights *= np.exp(-estimator_weight * y_transformed * y_pred)
            sample_weights /= np.sum(sample_weights)  # 归一化
            
            # 保存分类器和权重
            self.estimators.append(estimator)
            self.estimator_weights.append(estimator_weight)
            
    def predict(self, X):
        """
        预测样本类别
        :param X: 样本数据
        :return: 预测结果（0或1）
        """
        if not self.estimators:
            raise RuntimeError("模型尚未训练，请先调用train方法")
            
        X = np.array(X)
        # 初始化预测结果
        final_pred = np.zeros(X.shape[0])
        
        # 累加所有弱分类器的加权预测
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            y_pred = estimator.predict(X)
            final_pred += weight * y_pred
            
        # 将结果转换为0和1
        return np.where(np.sign(final_pred) == 1, 1, 0)
        
    def get_visualization_data(self):
        """获取AdaBoost可视化数据"""
        if not self.estimators:
            return None
            
        # 返回弱分类器信息和权重
        return {
            'n_estimators': self.n_estimators,
            'estimator_weights': self.estimator_weights,
            'estimator_count': len(self.estimators)
        }
