import numpy as np

class SVM:
    """支持向量机算法实现（二分类）"""
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, kernel='linear'):
        """
        初始化SVM模型
        :param learning_rate: 学习率
        :param lambda_param: 正则化参数
        :param n_iters: 迭代次数
        :param kernel: 核函数，目前仅支持'linear'
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.kernel = kernel
        self.w = None
        self.b = None
        self.support_vectors = None
        
    def _kernel_function(self, X1, X2):
        """核函数实现"""
        if self.kernel == 'linear':
            return np.dot(X1, X2)
        # 可以扩展其他核函数，如RBF、多项式等
        else:
            raise ValueError("不支持的核函数，目前仅支持'linear'")
            
    def train(self, X, y):
        """
        训练SVM模型
        :param X: 特征数据
        :param y: 标签数据（应为-1和1）
        """
        X = np.array(X)
        y = np.array(y)
        
        # 确保标签是-1和1
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        
        # 初始化权重和偏置
        self.w = np.zeros(n_features)
        self.b = 0
        
        # 梯度下降优化
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # 检查是否满足KKT条件
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                
                if condition:
                    # 对于支持向量外的点，只更新正则项
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    # 对于支持向量，更新权重和偏置
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]
        
        # 确定支持向量（满足y*(wx - b) <= 1的样本）
        support_vector_indices = []
        for idx, x_i in enumerate(X):
            if y_[idx] * (np.dot(x_i, self.w) - self.b) <= 1.001:  # 允许微小误差
                support_vector_indices.append(idx)
                
        self.support_vectors = {
            'X': X[support_vector_indices],
            'y': y[support_vector_indices]
        }
        
    def predict(self, X):
        """
        预测样本类别
        :param X: 样本数据
        :return: 预测结果（0或1）
        """
        if self.w is None:
            raise RuntimeError("模型尚未训练，请先调用train方法")
            
        X = np.array(X)
        linear_output = np.dot(X, self.w) - self.b
        # 将结果转换为0和1
        return np.where(np.sign(linear_output) == 1, 1, 0)
        
    def get_visualization_data(self):
        """获取SVM可视化数据"""
        if self.w is None:
            return None
            
        return {
            'weights': self.w.tolist(),
            'bias': self.b,
            'support_vectors': {
                'X': self.support_vectors['X'].tolist(),
                'y': self.support_vectors['y'].tolist()
            },
            'kernel': self.kernel
        }


