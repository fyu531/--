import math
import numpy as np

class LinearRegression:
    """线性回归模型"""
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=None, lambda_param=0.01):
        # 学习率
        self.learning_rate = learning_rate
        # 迭代次数
        self.n_iterations = n_iterations
        # 正则化类型 ('l1', 'l2' 或 None)
        self.regularization = regularization
        # 正则化参数
        self.lambda_param = lambda_param
        # 权重
        self.weights = None
        # 偏置
        self.bias = None
    
    def fit(self, features, labels):
        """训练线性回归模型"""
        if len(features) != len(labels):
            raise ValueError("特征和标签的数量必须相同")
        
        if len(features) == 0:
            raise ValueError("数据集不能为空")
        
        n_samples, n_features = len(features), len(features[0])
        
        # 初始化权重和偏置
        self.weights = [0.0 for _ in range(n_features)]
        self.bias = 0.0
        
        # 梯度下降
        for _ in range(self.n_iterations):
            # 计算预测值
            y_pred = [self._predict_sample(x) for x in features]
            
            # 计算权重梯度
            dw = [0.0 for _ in range(n_features)]
            for i in range(n_samples):
                error = y_pred[i] - labels[i]
                for j in range(n_features):
                    dw[j] += error * features[i][j]
            
            # 平均梯度
            for j in range(n_features):
                dw[j] /= n_samples
                
                # 添加正则化项
                if self.regularization == 'l2':
                    dw[j] += (self.lambda_param / n_samples) * self.weights[j]
                elif self.regularization == 'l1':
                    dw[j] += (self.lambda_param / n_samples) * (1 if self.weights[j] > 0 else -1)
            
            # 计算偏置梯度
            db = sum(y_pred[i] - labels[i] for i in range(n_samples)) / n_samples
            
            # 更新权重和偏置
            for j in range(n_features):
                self.weights[j] -= self.learning_rate * dw[j]
            self.bias -= self.learning_rate * db
    
    def _predict_sample(self, sample):
        """预测单个样本"""
        return sum(self.weights[j] * sample[j] for j in range(len(sample))) + self.bias
    
    def predict(self, features):
        """预测多个样本"""
        if self.weights is None or self.bias is None:
            raise ValueError("线性回归模型尚未训练，请先调用fit方法")
        
        return [self._predict_sample(sample) for sample in features]
