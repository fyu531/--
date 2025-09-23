import numpy as np
from scipy.stats import multivariate_normal  # 仅用于概率密度计算

class EMAlgorithm:
    """EM算法实现（高斯混合模型）"""
    def __init__(self, n_components=2, max_iter=100, tol=1e-4):
        """
        初始化EM算法
        :param n_components: 混合成分数量
        :param max_iter: 最大迭代次数
        :param tol: 收敛阈值
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        
        # 模型参数
        self.weights = None    # 混合权重
        self.means = None      # 均值
        self.covariances = None # 协方差矩阵
        self.log_likelihood = None  # 对数似然值
        
    def _initialize_parameters(self, X):
        """初始化模型参数"""
        n_samples, n_features = X.shape
        
        # 初始化混合权重（均匀分布）
        self.weights = np.ones(self.n_components) / self.n_components
        
        # 随机选择样本作为初始均值
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[indices]
        
        # 初始化协方差矩阵（对角矩阵）
        self.covariances = np.array([np.cov(X.T) for _ in range(self.n_components)])
        # 确保协方差矩阵是正定的
        for i in range(self.n_components):
            self.covariances[i] += np.eye(n_features) * 1e-6
            
    def _e_step(self, X):
        """E步：计算隐变量的后验概率"""
        n_samples = X.shape[0]
        # 责任矩阵：responsibilities[i][k]表示第i个样本属于第k个成分的概率
        responsibilities = np.zeros((n_samples, self.n_components))
        
        # 计算每个成分的概率密度
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights[k] * multivariate_normal.pdf(
                X, mean=self.means[k], cov=self.covariances[k]
            )
            
        # 归一化，得到后验概率
        total = np.sum(responsibilities, axis=1, keepdims=True)
        responsibilities /= total
        
        return responsibilities
        
    def _m_step(self, X, responsibilities):
        """M步：更新模型参数"""
        n_samples, n_features = X.shape
        
        # 计算每个成分的有效样本数
        n_k = np.sum(responsibilities, axis=0)
        
        # 更新混合权重
        self.weights = n_k / n_samples
        
        # 更新均值
        self.means = np.zeros((self.n_components, n_features))
        for k in range(self.n_components):
            self.means[k] = np.sum(responsibilities[:, k].reshape(-1, 1) * X, axis=0) / n_k[k]
            
        # 更新协方差矩阵
        self.covariances = np.zeros((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covariances[k] = np.dot(responsibilities[:, k] * diff.T, diff) / n_k[k]
            # 确保协方差矩阵是正定的
            self.covariances[k] += np.eye(n_features) * 1e-6
            
    def _compute_log_likelihood(self, X):
        """计算对数似然值"""
        n_samples = X.shape[0]
        log_likelihood = 0
        
        for i in range(n_samples):
            # 计算混合分布的概率密度
            total = 0
            for k in range(self.n_components):
                total += self.weights[k] * multivariate_normal.pdf(
                    X[i], mean=self.means[k], cov=self.covariances[k]
                )
            log_likelihood += np.log(total)
            
        return log_likelihood
        
    def train(self, X):
        """
        训练EM模型
        :param X: 特征数据（无标签）
        """
        X = np.array(X)
        
        # 初始化参数
        self._initialize_parameters(X)
        
        # 迭代EM步骤
        self.log_likelihood = []
        for _ in range(self.max_iter):
            # E步
            responsibilities = self._e_step(X)
            
            # M步
            self._m_step(X, responsibilities)
            
            # 计算对数似然值
            ll = self._compute_log_likelihood(X)
            self.log_likelihood.append(ll)
            
            # 检查是否收敛
            if len(self.log_likelihood) > 1 and \
               np.abs(self.log_likelihood[-1] - self.log_likelihood[-2]) < self.tol:
                break
                
    def predict(self, X):
        """
        预测样本所属成分
        :param X: 样本数据
        :return: 成分索引
        """
        if self.weights is None:
            raise RuntimeError("模型尚未训练，请先调用train方法")
            
        X = np.array(X)
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)
        
    def get_visualization_data(self):
        """获取EM算法可视化数据"""
        if self.weights is None:
            return None
            
        return {
            'n_components': self.n_components,
            'weights': self.weights.tolist(),
            'means': self.means.tolist(),
            'log_likelihood': self.log_likelihood
        }
