import numpy as np

class LinearRegression:
    """线性回归模型（修复数值溢出和稳定性问题）"""
    def __init__(self, learning_rate=0.001, n_iterations=1000, regularization=None, 
                 lambda_param=0.01, normalize=True):
        self.learning_rate = learning_rate  # 减小默认学习率
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None
        self.train_mse_history = []
        self.normalize = normalize  # 新增：是否归一化特征
        self.scaler_mean = None     # 用于存储特征均值（归一化用）
        self.scaler_std = None      # 用于存储特征标准差（归一化用）
    
    def fit(self, features, labels):
        """训练模型并记录MSE变化，添加特征归一化"""
        if len(features) != len(labels):
            raise ValueError("特征和标签的数量必须相同")
        
        if len(features) == 0:
            raise ValueError("数据集不能为空")
        
        # 确保输入是numpy数组且标签为一维
        features = np.array(features, dtype=np.float64)
        labels = np.array(labels, dtype=np.float64).flatten()
        
        # 特征归一化处理
        if self.normalize:
            self.scaler_mean = np.mean(features, axis=0)
            self.scaler_std = np.std(features, axis=0) + 1e-8  # 避免除以零
            features = (features - self.scaler_mean) / self.scaler_std
        
        n_samples, n_features = features.shape
        
        # 初始化参数（使用更小的初始值）
        self.weights = np.zeros(n_features) * 0.01  # 更小的初始权重
        self.bias = 0.0
        self.train_mse_history = []
        
        # 梯度下降
        for _ in range(self.n_iterations):
            y_pred = self._predict(features)
            
            # 计算并记录MSE
            mse = float(np.mean((labels - y_pred) **2))
            self.train_mse_history.append(mse)
            
            # 检查是否出现NaN或无穷大，及时终止训练
            if np.isnan(mse) or np.isinf(mse):
                print(f"在迭代 {_} 时出现数值不稳定，终止训练")
                break
            
            # 计算梯度
            error = y_pred - labels
            dw = (1 / n_samples) * np.dot(features.T, error)
            db = (1 / n_samples) * np.sum(error)
            
            # 添加正则化
            if self.regularization == 'l2':
                dw += (self.lambda_param / n_samples) * self.weights
            elif self.regularization == 'l1':
                dw += (self.lambda_param / n_samples) * np.sign(self.weights)
            
            # 梯度裁剪，防止梯度爆炸
            gradient_norm = np.linalg.norm(dw)
            if gradient_norm > 1.0:  # 设定阈值
                dw = dw / gradient_norm
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def train(self, features, labels):
        """适配后端接口的训练方法"""
        self.fit(features, labels)
    
    def _predict(self, features):
        """内部预测函数（返回numpy数组用于计算）"""
        # 如果启用了归一化，预测时也要对输入特征进行归一化
        if self.normalize and self.scaler_mean is not None:
            features = (features - self.scaler_mean) / (self.scaler_std + 1e-8)
        return np.dot(features, self.weights) + self.bias
    
    def predict(self, features):
        """预测接口（返回Python列表用于前端展示）"""
        if self.weights is None or self.bias is None:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        features = np.array(features, dtype=np.float64)
        return self._predict(features).tolist()
    
    def predict_np(self, features):
        """预测接口（返回numpy数组用于计算MSE）"""
        if self.weights is None or self.bias is None:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        features = np.array(features, dtype=np.float64)
        return self._predict(features)
    
    def get_metrics(self, X, y_true):
        """计算并返回评估指标"""
        y_pred = self.predict_np(X)
        
        # 计算MSE
        mse = float(np.mean((y_true - y_pred) **2))
        # 计算RMSE
        rmse = float(np.sqrt(mse)) if mse >= 0 else np.nan
        # 计算MAE
        mae = float(np.mean(np.abs(y_true - y_pred)))
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }
    
    def get_visualization_data(self):
        """返回可视化数据"""
        coefficients = [float(w) for w in self.weights] if self.weights is not None else []
        intercept = float(self.bias) if self.bias is not None else 0.0
        
        return {
            'type': 'linear_regression',
            'coefficients': coefficients,
            'intercept': intercept,
            'learning_rate': float(self.learning_rate),
            'n_iterations': int(self.n_iterations),
            'train_mse_history': self.train_mse_history or [],
            'regularization': self.regularization or 'none',
            'normalize': self.normalize
        }


