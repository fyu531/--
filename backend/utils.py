import math
import random
from collections import defaultdict
import numpy as np

def train_test_split(data, test_size=0.3, random_state=None):
    """
    将数据集分割为训练集和测试集
    
    参数:
        data: 数据集，包含'features'和'labels'键
        test_size: 测试集比例
        random_state: 随机种子，用于重现结果
    
    返回:
        训练集和测试集，每个都是包含'features'和'labels'的字典
    """
    if random_state is not None:
        random.seed(random_state)
    
    # 生成索引并打乱
    indices = list(range(len(data['features'])))
    random.shuffle(indices)
    
    # 分割索引
    split_index = int(len(indices) * (1 - test_size))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    
    # 创建训练集和测试集
    train_features = [data['features'][i] for i in train_indices]
    train_labels = [data['labels'][i] for i in train_indices]
    test_features = [data['features'][i] for i in test_indices]
    test_labels = [data['labels'][i] for i in test_indices]
    
    return {
        'train': {'features': train_features, 'labels': train_labels},
        'test': {'features': test_features, 'labels': test_labels}
    }

def accuracy_score(y_true, y_pred):
    """计算准确率"""
    if len(y_true) != len(y_pred):
        raise ValueError("真实标签和预测标签长度必须相同")
    
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

def precision_score(y_true, y_pred, average='macro'):
    """计算精确率"""
    if len(y_true) != len(y_pred):
        raise ValueError("真实标签和预测标签长度必须相同")
    
    classes = list(set(y_true))
    precision = {}
    
    for cls in classes:
        # 真正例
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred == cls)
        # 假正例
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != cls and pred == cls)
        
        if tp + fp == 0:
            precision[cls] = 0.0
        else:
            precision[cls] = tp / (tp + fp)
    
    if average == 'macro':
        return sum(precision.values()) / len(precision)
    elif average == 'micro':
        tp_total = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        fp_total = sum(1 for true, pred in zip(y_true, y_pred) if true != pred)
        return tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    elif average is None:
        return precision
    else:
        raise ValueError(f"不支持的平均方式: {average}")

def recall_score(y_true, y_pred, average='macro'):
    """计算召回率"""
    if len(y_true) != len(y_pred):
        raise ValueError("真实标签和预测标签长度必须相同")
    
    classes = list(set(y_true))
    recall = {}
    
    for cls in classes:
        # 真正例
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred == cls)
        # 假负例
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred != cls)
        
        if tp + fn == 0:
            recall[cls] = 0.0
        else:
            recall[cls] = tp / (tp + fn)
    
    if average == 'macro':
        return sum(recall.values()) / len(recall)
    elif average == 'micro':
        tp_total = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        fn_total = sum(1 for true, pred in zip(y_true, y_pred) if true != pred)
        return tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    elif average is None:
        return recall
    else:
        raise ValueError(f"不支持的平均方式: {average}")

def f1_score(y_true, y_pred, average='macro'):
    """计算F1分数"""
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    
    f1 = {}
    for cls in precision:
        if precision[cls] + recall[cls] == 0:
            f1[cls] = 0.0
        else:
            f1[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls])
    
    if average == 'macro':
        return sum(f1.values()) / len(f1)
    elif average == 'micro':
        tp_total = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        fp_total = sum(1 for true, pred in zip(y_true, y_pred) if true != pred)
        fn_total = fp_total  # 简化实现
        if tp_total + (fp_total + fn_total) / 2 == 0:
            return 0.0
        return 2 * tp_total / (2 * tp_total + fp_total + fn_total)
    elif average is None:
        return f1
    else:
        raise ValueError(f"不支持的平均方式: {average}")

def mean_squared_error(y_true, y_pred):
    """计算均方误差"""
    if len(y_true) != len(y_pred):
        raise ValueError("真实值和预测值长度必须相同")
    
    return sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred)) / len(y_true)

def normalize_features(features):
    """标准化特征，使每个特征的均值为0，标准差为1"""
    if not features or not features[0]:
        return features
    
    num_samples = len(features)
    num_features = len(features[0])
    
    # 计算每个特征的均值和标准差
    means = []
    stds = []
    
    for i in range(num_features):
        feature_values = [features[j][i] for j in range(num_samples)]
        mean = sum(feature_values) / num_samples
        std = math.sqrt(sum((x - mean) **2 for x in feature_values) / num_samples)
        
        # 避免除以零
        if std < 1e-10:
            std = 1.0
        
        means.append(mean)
        stds.append(std)
    
    # 标准化特征
    normalized = []
    for sample in features:
        normalized_sample = [(sample[i] - means[i]) / stds[i] for i in range(num_features)]
        normalized.append(normalized_sample)
    
    return normalized

def euclidean_distance(x1, x2):
    """计算欧氏距离"""
    if len(x1) != len(x2):
        raise ValueError("两个向量的维度必须相同")
    
    return math.sqrt(sum((a - b)** 2 for a, b in zip(x1, x2)))

def manhattan_distance(x1, x2):
    """计算曼哈顿距离"""
    if len(x1) != len(x2):
        raise ValueError("两个向量的维度必须相同")
    
    return sum(abs(a - b) for a, b in zip(x1, x2))

# majority_vote 函数
def majority_vote(labels):
    import numpy as np
    counts = np.bincount(labels)
    return np.argmax(counts)

def entropy(labels):
    """计算熵"""
    if not labels:
        return 0.0
    
    label_counts = defaultdict(int)
    for label in labels:
        label_counts[label] += 1
    
    entropy = 0.0
    total = len(labels)
    
    for count in label_counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    
    return entropy

def gini_impurity(labels):
    """计算基尼不纯度"""
    if not labels:
        return 0.0
    
    label_counts = defaultdict(int)
    for label in labels:
        label_counts[label] += 1
    
    impurity = 1.0
    total = len(labels)
    
    for count in label_counts.values():
        p = count / total
        impurity -= p **2
    
    return impurity

# split_dataset 函数
def split_dataset(features, labels, feature_idx, threshold):
    import numpy as np
    features = np.array(features)
    labels = np.array(labels)
    
    mask = features[:, feature_idx] < threshold
    left_features = features[mask].tolist()
    left_labels = labels[mask].tolist()
    right_features = features[~mask].tolist()
    right_labels = labels[~mask].tolist()
    
    return left_features, left_labels, right_features, right_labels

def one_hot_encode(labels):
    """对标签进行独热编码"""
    unique_labels = list(set(labels))
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    
    encoded = []
    for label in labels:
        vec = [0] * len(unique_labels)
        vec[label_to_index[label]] = 1
        encoded.append(vec)
    
    return encoded, unique_labels
