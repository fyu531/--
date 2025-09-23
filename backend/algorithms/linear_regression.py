import numpy as np

class LinearRegression:
    """线性回归模型（修复MSE加载问题）"""
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=None, lambda_param=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None
        self.train_mse_history = []  # 训练过程的MSE记录
    
    def fit(self, features, labels):
        """训练模型并记录MSE变化"""
        if len(features) != len(labels):
            raise ValueError("特征和标签的数量必须相同")
        
        if len(features) == 0:
            raise ValueError("数据集不能为空")
        
        # 确保输入是numpy数组且标签为一维
        features = np.array(features, dtype=np.float64)
        labels = np.array(labels, dtype=np.float64).flatten()
        
        n_samples, n_features = features.shape
        
        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.train_mse_history = []
        
        # 梯度下降
        for _ in range(self.n_iterations):
            y_pred = self._predict(features)
            
            # 计算并记录MSE（转为Python类型）
            mse = float(np.mean((labels - y_pred) **2))
            self.train_mse_history.append(mse)
            
            # 计算梯度
            error = y_pred - labels
            dw = (1 / n_samples) * np.dot(features.T, error)
            db = (1 / n_samples) * np.sum(error)
            
            # 添加正则化
            if self.regularization == 'l2':
                dw += (self.lambda_param / n_samples) * self.weights
            elif self.regularization == 'l1':
                dw += (self.lambda_param / n_samples) * np.sign(self.weights)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def train(self, features, labels):
        """适配后端接口的训练方法"""
        self.fit(features, labels)
    
    def _predict(self, features):
        """内部预测函数（返回numpy数组用于计算）"""
        return np.dot(features, self.weights) + self.bias
    
    def predict(self, features):
        """预测接口（返回Python列表用于前端展示）"""
        if self.weights is None or self.bias is None:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        features = np.array(features, dtype=np.float64)
        return self._predict(features).tolist()  # 转为Python列表确保序列化
    
    def predict_np(self, features):
        """预测接口（返回numpy数组用于计算MSE）"""
        if self.weights is None or self.bias is None:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        features = np.array(features, dtype=np.float64)
        return self._predict(features)
    
    def get_metrics(self, X, y_true):
        """计算并返回评估指标（确保所有值都是Python类型）"""
        y_pred = self.predict_np(X)
        
        # 计算MSE
        mse = float(np.mean((y_true - y_pred) **2))
        # 计算RMSE
        rmse = float(np.sqrt(mse))
        # 计算MAE
        mae = float(np.mean(np.abs(y_true - y_pred)))
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }
    
    def get_visualization_data(self):
        """返回可视化数据（确保所有值都是JSON可序列化类型）"""
        return {
            'type': 'linear_regression',
            'coefficients': [float(w) for w in self.weights] if self.weights is not None else [],
            'intercept': float(self.bias) if self.bias is not None else 0.0,
            'learning_rate': float(self.learning_rate),
            'n_iterations': int(self.n_iterations),
            'train_mse_history': self.train_mse_history,  # 已确保是Python列表
            'regularization': self.regularization
        }

