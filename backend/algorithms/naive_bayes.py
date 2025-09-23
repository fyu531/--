import math
from collections import defaultdict
from backend.utils import one_hot_encode

class NaiveBayes:
    """朴素贝叶斯算法实现（适用于分类问题）"""
    def __init__(self, laplace_smoothing=1e-9):
        self.laplace_smoothing = laplace_smoothing  # 拉普拉斯平滑参数
        self.class_priors = {}                      # 类先验概率 P(y)
        self.feature_probs = defaultdict(dict)      # 特征条件概率 P(x_i|y)
        self.classes = None                         # 所有类别
        self.num_features = None                    # 特征数量
    
    def train(self, features, labels):
        """训练朴素贝叶斯模型"""
        if len(features) == 0:
            raise ValueError("训练数据不能为空")
        if len(features) != len(labels):
            raise ValueError("特征和标签数量必须相同")
        
        self.num_features = len(features[0])
        self.classes = list(set(labels))
        num_samples = len(features)
        
        # 计算类先验概率 P(y)
        class_counts = defaultdict(int)
        for label in labels:
            class_counts[label] += 1
        
        for cls in self.classes:
            self.class_priors[cls] = (class_counts[cls] + self.laplace_smoothing) / \
                                    (num_samples + self.laplace_smoothing * len(self.classes))
        
        # 计算每个特征的条件概率 P(x_i|y)
        # 首先按类别分组特征
        class_features = defaultdict(list)
        for feature, label in zip(features, labels):
            class_features[label].append(feature)
        
        # 对每个类别和每个特征计算条件概率
        for cls in self.classes:
            features_for_cls = class_features[cls]
            num_samples_cls = len(features_for_cls)
            
            for feature_idx in range(self.num_features):
                # 收集该类别下该特征的所有值
                feature_values = [f[feature_idx] for f in features_for_cls]
                
                # 计算每个可能值的概率（假设特征是离散的）
                value_counts = defaultdict(int)
                for value in feature_values:
                    # 为了处理连续值，我们将其离散化
                    # 这里简单地四舍五入到小数点后两位
                    rounded_value = round(value, 2)
                    value_counts[rounded_value] += 1
                
                # 计算每个值的条件概率（带拉普拉斯平滑）
                total_values = len(value_counts)
                for value, count in value_counts.items():
                    prob = (count + self.laplace_smoothing) / \
                          (num_samples_cls + self.laplace_smoothing * (total_values + 1))
                    self.feature_probs[(cls, feature_idx)][value] = prob
    
    def _predict_sample(self, sample):
        """预测单个样本"""
        # 计算每个类别的后验概率（对数形式，避免下溢）
        log_probs = {}
        
        for cls in self.classes:
            # 先验概率的对数
            log_prob = math.log(self.class_priors[cls])
            
            # 加上每个特征的条件概率的对数
            for feature_idx in range(self.num_features):
                value = round(sample[feature_idx], 2)
                
                # 如果该值在训练集中没有出现过，使用平滑后的概率
                if value not in self.feature_probs[(cls, feature_idx)]:
                    total_values = len(self.feature_probs[(cls, feature_idx)])
                    prob = self.laplace_smoothing / \
                          (sum(self.feature_probs[(cls, feature_idx)].values()) * len(self.classes) + 
                           self.laplace_smoothing * (total_values + 1))
                    log_prob += math.log(prob)
                else:
                    log_prob += math.log(self.feature_probs[(cls, feature_idx)][value])
            
            log_probs[cls] = log_prob
        
        # 返回后验概率最大的类别
        return max(log_probs, key=log_probs.get)
    
    def predict(self, features):
        """预测多个样本"""
        if self.classes is None:
            raise RuntimeError("朴素贝叶斯模型尚未训练，请先调用train方法")
        
        return [self._predict_sample(sample) for sample in features]
    
    def get_visualization_data(self):
        """获取朴素贝叶斯可视化数据"""
        if self.classes is None:
            return None
            
        # 返回先验概率和部分条件概率用于可视化
        visualization_data = {
            'class_priors': self.class_priors,
            'feature_probs_sample': {}  # 只取部分特征概率，避免数据过大
        }
        
        # 每个类别取前2个特征的概率分布
        for cls in self.classes[:2]:  # 限制类别数量
            for feature_idx in range(min(2, self.num_features)):  # 限制特征数量
                if (cls, feature_idx) in self.feature_probs:
                    # 只取前5个值的概率
                    probs = list(self.feature_probs[(cls, feature_idx)].items())[:5]
                    visualization_data['feature_probs_sample'][(cls, feature_idx)] = dict(probs)
        
        return visualization_data
