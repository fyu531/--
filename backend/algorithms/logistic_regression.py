import math
import numpy as np

class LogisticRegression:
    """逻辑回归分类器（支持二分类和多分类）"""
    def __init__(self, learning_rate=0.01, n_iterations=1000, 
                 regularization=None, lambda_param=0.01, 
                 multi_class='ovr'):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_param = lambda_param
        self.multi_class = multi_class  # 'ovr'表示One-vs-Rest多分类
        self.weights = None
        self.bias = None
        self.class_mapping = None
        self.classifiers = None  # 存储多分类器
    
    def _sigmoid(self, z):
        """Sigmoid函数"""
        if isinstance(z, (int, float)):
            if z >= 0:
                return 1 / (1 + math.exp(-z))
            else:
                return math.exp(z) / (1 + math.exp(z))
        
        # 处理数组情况
        z = np.clip(z, -500, 500)  # 防止溢出
        return 1.0 / (1.0 + np.exp(-z))
    
    def fit(self, features, labels):
        """训练逻辑回归模型"""
        features = np.array(features, dtype=np.float64)
        labels = np.array(labels).flatten()
        
        if features.shape[0] != labels.shape[0]:
            raise ValueError("特征和标签的数量必须相同")
        
        if features.shape[0] == 0:
            raise ValueError("数据集不能为空")
        
        classes = np.unique(labels)
        self.class_mapping = {cls: i for i, cls in enumerate(classes)}
        
        # 二分类
        if len(classes) == 2:
            y = np.array([self.class_mapping[label] for label in labels], dtype=np.float64)
            self._fit_binary(features, y)
        # 多分类
        else:
            if self.multi_class == 'ovr':
                self._fit_ovr(features, labels, classes)
            else:
                raise ValueError(f"不支持的多分类策略: {self.multi_class}")
    
    def _fit_binary(self, features, y):
        """二分类训练"""
        n_samples, n_features = features.shape
        
        # 初始化参数
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0
        
        # 梯度下降
        for _ in range(self.n_iterations):
            y_pred_proba = self._predict_proba_np(features)
            
            # 计算梯度
            error = y_pred_proba - y
            dw = (1 / n_samples) * np.dot(features.T, error)
            db = (1 / n_samples) * np.sum(error)
            
            # 正则化
            if self.regularization == 'l2':
                dw += (self.lambda_param / n_samples) * self.weights
            elif self.regularization == 'l1':
                dw += (self.lambda_param / n_samples) * np.sign(self.weights)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def _fit_ovr(self, features, labels, classes):
        """One-vs-Rest多分类训练"""
        self.classifiers = []
        
        for cls in classes:
            # 创建新分类器
            clf = LogisticRegression(
                learning_rate=self.learning_rate,
                n_iterations=self.n_iterations,
                regularization=self.regularization,
                lambda_param=self.lambda_param
            )
            
            # 构建二分类问题
            binary_labels = np.where(labels == cls, 1, 0)
            
            # 训练
            clf.fit(features, binary_labels)
            self.classifiers.append(clf)
    
    def _predict_proba_np(self, features):
        """内部预测概率（numpy版本）"""
        z = np.dot(features, self.weights) + self.bias
        return self._sigmoid(z)
    
    def predict(self, features, threshold=0.5):
        """预测多个样本的类别"""
        features = np.array(features, dtype=np.float64)
        
        if self.classifiers is not None:  # 多分类
            probabilities = []
            for clf in self.classifiers:
                prob = clf.predict_proba(features)
                probabilities.append(prob)
            
            probabilities = np.array(probabilities).T
            class_indices = np.argmax(probabilities, axis=1)
            
            # 映射回原始类别
            inverse_mapping = {i: cls for cls, i in self.class_mapping.items()}
            return [inverse_mapping[idx] for idx in class_indices]
        else:  # 二分类
            if self.weights is None or self.bias is None:
                raise ValueError("逻辑回归模型尚未训练，请先调用fit或train方法")
            
            y_pred_proba = self.predict_proba(features)
            inverse_mapping = {v: k for k, v in self.class_mapping.items()}
            return [inverse_mapping[1] if p >= threshold else inverse_mapping[0] 
                    for p in y_pred_proba]
    
    def predict_proba(self, features):
        """预测多个样本属于各类别的概率"""
        features = np.array(features, dtype=np.float64)
        
        if self.classifiers is not None:  # 多分类
            probabilities = []
            for clf in self.classifiers:
                prob = clf.predict_proba(features)
                probabilities.append(prob)
            
            probabilities = np.array(probabilities).T
            return probabilities.tolist()
        else:  # 二分类
            if self.weights is None or self.bias is None:
                raise ValueError("逻辑回归模型尚未训练，请先调用fit或train方法")
                
            return self._predict_proba_np(features).tolist()
    
    def train(self, features, labels):
        """适配后端接口的训练方法"""
        self.fit(features, labels)
        
    def get_visualization_data(self):
        """返回可视化数据（确保所有值都是JSON可序列化类型）"""
        coefficients = []
        if self.classifiers is not None:  # 多分类
            # 每个分类器的权重
            coefficients = [clf.weights.tolist() if hasattr(clf, 'weights') and clf.weights is not None else [] 
                           for clf in self.classifiers]
        elif self.weights is not None:  # 二分类
            coefficients = self.weights.tolist()
        
        return {
            'type': 'logistic_regression',
            'coefficients': coefficients,
            'intercept': float(self.bias) if self.bias is not None else 0.0,
            'learning_rate': float(self.learning_rate),
            'n_iterations': int(self.n_iterations),
            'regularization': self.regularization or 'none',
            'multi_class': self.multi_class
        }