import math

class LogisticRegression:
    """逻辑回归分类器（二分类）"""
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
    
    def _sigmoid(self, z):
        """Sigmoid函数"""
        # 防止溢出
        if z >= 0:
            return 1 / (1 + math.exp(-z))
        else:
            return math.exp(z) / (1 + math.exp(z))
    
    def fit(self, features, labels):
        """训练逻辑回归模型"""
        if len(features) != len(labels):
            raise ValueError("特征和标签的数量必须相同")
        
        if len(features) == 0:
            raise ValueError("数据集不能为空")
        
        # 检查是否为二分类问题
        classes = set(labels)
        if len(classes) != 2:
            raise ValueError("逻辑回归目前只支持二分类问题")
        
        # 将标签转换为0和1
        self.class_mapping = {cls: i for i, cls in enumerate(classes)}
        y = [self.class_mapping[label] for label in labels]
        
        n_samples, n_features = len(features), len(features[0])
        
        # 初始化权重和偏置
        self.weights = [0.0 for _ in range(n_features)]
        self.bias = 0.0
        
        # 梯度下降
        for _ in range(self.n_iterations):
            # 计算预测概率
            y_pred_proba = [self._sigmoid(sum(self.weights[j] * x[j] for j in range(n_features)) + self.bias) for x in features]
            
            # 计算权重梯度
            dw = [0.0 for _ in range(n_features)]
            for i in range(n_samples):
                error = y_pred_proba[i] - y[i]
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
            db = sum(y_pred_proba[i] - y[i] for i in range(n_samples)) / n_samples
            
            # 更新权重和偏置
            for j in range(n_features):
                self.weights[j] -= self.learning_rate * dw[j]
            self.bias -= self.learning_rate * db
    
    def _predict_proba_sample(self, sample):
        """预测单个样本属于正类的概率"""
        z = sum(self.weights[j] * sample[j] for j in range(len(sample))) + self.bias
        return self._sigmoid(z)
    
    def predict(self, features, threshold=0.5):
        """预测多个样本的类别"""
        if self.weights is None or self.bias is None:
            raise ValueError("逻辑回归模型尚未训练，请先调用fit方法")
        
        # 预测概率
        y_pred_proba = [self._predict_proba_sample(sample) for sample in features]
        
        # 根据阈值转换为类别
        inverse_mapping = {v: k for k, v in self.class_mapping.items()}
        return [inverse_mapping[1] if p >= threshold else inverse_mapping[0] for p in y_pred_proba]
    
    def predict_proba(self, features):
        """预测多个样本属于正类的概率"""
        if self.weights is None or self.bias is None:
            raise ValueError("逻辑回归模型尚未训练，请先调用fit方法")
        
        return [self._predict_proba_sample(sample) for sample in features]
