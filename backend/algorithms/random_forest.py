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
        self.n_features = n_features
        self.trees = []
        
    def _bootstrap_sample(self, X, y):
        """生成bootstrap样本"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
        
    def train(self, X, y):
        """
        训练随机森林模型
        :param X: 特征数据
        :param y: 标签数据
        """
        X = np.array(X)
        y = np.array(y)
        
        self.trees = []
        n_features = X.shape[1] if self.n_features is None else min(self.n_features, X.shape[1])
        
        for _ in range(self.n_trees):
            # 创建决策树
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=n_features
            )
            
            # 生成bootstrap样本
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # 训练决策树
            tree.train(X_sample, y_sample)
            
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
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # 对预测结果进行多数投票
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [self._most_common_label(preds) for preds in tree_preds]
        
        return np.array(y_pred)
        
    def _most_common_label(self, y):
        """返回最常见的标签"""
        counts = np.bincount(y.astype(int))
        return np.argmax(counts)
        
    def get_visualization_data(self):
        """获取随机森林可视化数据"""
        if not self.trees:
            return None
            
        # 返回部分树的信息用于可视化
        sample_trees = min(3, self.n_trees)
        tree_data = []
        
        for i in range(sample_trees):
            tree_data.append({
                'depth': self.trees[i].get_depth(),
                'node_count': self.trees[i].get_node_count(),
                'feature_importance': self.trees[i].get_feature_importance()
            })
            
        return {
            'n_trees': self.n_trees,
            'sample_trees': tree_data,
            'max_depth': self.max_depth
        }
