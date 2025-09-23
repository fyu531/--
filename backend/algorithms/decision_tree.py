import math
import random
from collections import defaultdict
from backend.utils import entropy, gini_impurity, split_dataset, majority_vote

class DecisionTreeNode:
    """决策树节点"""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # 用于分割的特征索引
        self.threshold = threshold      # 分割阈值
        self.left = left                # 左子树（特征值 < 阈值）
        self.right = right              # 右子树（特征值 >= 阈值）
        self.value = value              # 叶节点的预测值

class DecisionTree:
    """决策树算法实现"""
    def __init__(self, max_depth=5, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth          # 树的最大深度
        self.min_samples_split = min_samples_split  # 最小分裂样本数
        self.criterion = criterion          # 不纯度计算标准：'gini' 或 'entropy'
        self.root = None                    # 根节点
        
    def _calculate_impurity(self, labels):
        """计算不纯度"""
        if self.criterion == 'gini':
            return gini_impurity(labels)
        elif self.criterion == 'entropy':
            return entropy(labels)
        else:
            raise ValueError(f"不支持的不纯度计算标准: {self.criterion}")
    
    def _find_best_split(self, features, labels):
        """寻找最佳分裂点"""
        best_gain = -1
        best_feature_idx = None
        best_threshold = None
        num_samples, num_features = len(features), len(features[0]) if features else 0
        
        # 计算父节点的不纯度
        parent_impurity = self._calculate_impurity(labels)
        
        # 遍历每个特征
        for feature_idx in range(num_features):
            # 获取该特征的所有值
            feature_values = [features[i][feature_idx] for i in range(num_samples)]
            thresholds = list(set(feature_values))  # 唯一值作为可能的阈值
            
            # 遍历每个可能的阈值
            for threshold in thresholds:
                # 分割数据集
                left_features, left_labels, right_features, right_labels = split_dataset(
                    features, labels, feature_idx, threshold
                )
                
                # 如果分割后某一子集为空，则跳过
                if len(left_labels) == 0 or len(right_labels) == 0:
                    continue
                
                # 计算信息增益
                left_impurity = self._calculate_impurity(left_labels)
                right_impurity = self._calculate_impurity(right_labels)
                
                # 加权不纯度
                weighted_impurity = (len(left_labels) / num_samples) * left_impurity + \
                                   (len(right_labels) / num_samples) * right_impurity
                
                # 信息增益 = 父节点不纯度 - 子节点加权不纯度
                gain = parent_impurity - weighted_impurity
                
                # 更新最佳分裂点
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        return best_feature_idx, best_threshold, best_gain
    
    def _build_tree(self, features, labels, depth=0):
        """递归构建决策树"""
        num_samples = len(features)
        num_unique_labels = len(set(labels))
        
        # 终止条件
        if (depth >= self.max_depth or 
            num_samples < self.min_samples_split or 
            num_unique_labels == 1):
            # 创建叶节点，值为多数标签
            leaf_value = majority_vote(labels)
            return DecisionTreeNode(value=leaf_value)
        
        # 寻找最佳分裂点
        best_feature_idx, best_threshold, best_gain = self._find_best_split(features, labels)
        
        # 如果没有找到有意义的分裂点，创建叶节点
        if best_gain <= 0:
            leaf_value = majority_vote(labels)
            return DecisionTreeNode(value=leaf_value)
        
        # 分割数据集
        left_features, left_labels, right_features, right_labels = split_dataset(
            features, labels, best_feature_idx, best_threshold
        )
        
        # 递归构建左右子树
        left_subtree = self._build_tree(left_features, left_labels, depth + 1)
        right_subtree = self._build_tree(right_features, right_labels, depth + 1)
        
        # 返回当前节点
        return DecisionTreeNode(
            feature_idx=best_feature_idx,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree
        )
    
    def train(self, features, labels):
        """训练决策树"""
        if len(features) == 0:
            raise ValueError("训练数据不能为空")
        if len(features) != len(labels):
            raise ValueError("特征和标签数量必须相同")
        
        self.root = self._build_tree(features, labels)
    
    def _predict_sample(self, sample, node):
        """预测单个样本"""
        # 如果是叶节点，返回其值
        if node.value is not None:
            return node.value
        
        # 否则递归预测
        feature_value = sample[node.feature_idx]
        if feature_value < node.threshold:
            return self._predict_sample(sample, node.left)
        else:
            return self._predict_sample(sample, node.right)
    
    def predict(self, features):
        """预测多个样本"""
        if self.root is None:
            raise RuntimeError("决策树尚未训练，请先调用train方法")
        
        return [self._predict_sample(sample, self.root) for sample in features]
    
    def get_visualization_data(self):
        """获取决策树可视化数据"""
        if self.root is None:
            return None
            
        # 递归构建树的可视化结构
        def build_tree_data(node, depth=0):
            if node.value is not None:
                return {
                    'type': 'leaf',
                    'value': node.value,
                    'depth': depth
                }
            else:
                return {
                    'type': 'node',
                    'feature_idx': node.feature_idx,
                    'threshold': node.threshold,
                    'left': build_tree_data(node.left, depth + 1),
                    'right': build_tree_data(node.right, depth + 1),
                    'depth': depth
                }
        
        return build_tree_data(self.root)
