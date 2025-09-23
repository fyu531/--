import numpy as np
import random
from .decision_tree import DecisionTree

class RandomForest:
    """随机森林算法实现（分类）"""
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        """
        初始化随机森林模型
        :param n_trees: 树的数量
        :param max_depth: 树的最大深度
        :param min_samples_split: 最小分裂样本数
        :param n_features: 每次分裂考虑的特征数量
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features  # 每棵树使用的特征数量
        self.trees = []
        self.feature_indices_ = None  # 存储每棵树使用的特征索引
        
    def _bootstrap_sample(self, X, y):
        """生成bootstrap样本"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    # 确保train方法正确实现
    def train(self, X, y):
        """
        训练随机森林模型
        :param X: 特征数据
        :param y: 标签数据
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
            
        n_features = X.shape[1]
        
        # 确定每棵树使用的特征数量
        if self.n_features is None:
            self.n_features = int(np.sqrt(n_features))  # 分类问题默认使用sqrt(n_features)
        else:
            self.n_features = min(self.n_features, n_features)
        
        self.trees = []
        self.feature_indices_ = []  # 记录每棵树使用的特征
        
        for _ in range(self.n_trees):
            # 随机选择特征
            feature_indices = np.random.choice(n_features, self.n_features, replace=False)
            # 将numpy数组索引转换为Python列表，避免JSON序列化问题
            self.feature_indices_.append([int(idx) for idx in feature_indices])
            
            # 创建决策树
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            
            # 生成bootstrap样本
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # 只使用选中的特征
            X_sample_subset = X_sample[:, feature_indices]
            
            # 训练决策树（确保DecisionTree类有train方法）
            tree.train(X_sample_subset, y_sample)
            
            # 添加到森林
            self.trees.append(tree)
    
    def predict(self, X):
        """
        预测样本类别
        :param X: 样本数据
        :return: 预测结果
        """
        if not self.trees:
            raise RuntimeError("模型尚未训练，请先调用train方法")
            
        X = np.array(X)
        # 收集所有树的预测结果
        tree_preds = np.array([
            tree.predict(X[:, indices]) 
            for tree, indices in zip(self.trees, self.feature_indices_)
        ])
        
        # 对预测结果进行多数投票
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [self._most_common_label(preds) for preds in tree_preds]
        
        return np.array(y_pred)
    
    def evaluate(self, X, y):
        """评估模型性能"""
        X = np.array(X)
        y = np.array(y)
        y_pred = self.predict(X)
        
        # 计算准确率
        accuracy = np.mean(y_pred == y)
        
        return {
            'accuracy': float(accuracy)  # 确保转换为Python类型
        }
        
    def _most_common_label(self, y):
        """返回最常见的标签"""
        unique, counts = np.unique(y, return_counts=True)
        # 确保返回Python原生类型而非numpy类型
        return int(unique[np.argmax(counts)]) if isinstance(unique[0], np.integer) else unique[np.argmax(counts)]
        
    def get_visualization_data(self):
        """获取随机森林可视化数据，确保所有数据可JSON序列化"""
        if not self.trees:
            return None
            
        # 返回部分树的信息用于可视化
        sample_trees = min(3, self.n_trees)
        tree_data = []
        
        for i in range(sample_trees):
            # 获取决策树的可视化数据
            tree_vis = self.trees[i].get_visualization_data()
            
            # 确保特征重要性是Python类型
            feature_importance = {}
            if tree_vis and 'feature_importance' in tree_vis:
                for k, v in tree_vis['feature_importance'].items():
                    # 确保键是字符串或整数，值是Python浮点数
                    key = int(k) if isinstance(k, np.integer) else k
                    feature_importance[key] = float(v)
            
            tree_data.append({
                'depth': int(tree_vis['depth']) if tree_vis and 'depth' in tree_vis else 0,
                'node_count': int(tree_vis['node_count']) if tree_vis and 'node_count' in tree_vis else 0,
                'feature_indices': self.feature_indices_[i],  # 已经转换为Python列表
                'feature_importance': feature_importance
            })
            
        return {
            'n_trees': self.n_trees,
            'sample_trees': tree_data,
            'max_depth': self.max_depth,
            'n_features': self.n_features
        }
