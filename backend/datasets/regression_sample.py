import numpy as np

def load_regression_sample():
    """
    加载回归样本数据集
    包含多个特征和一个连续的目标值
    
    返回:
        X: 特征数据
        y: 目标值（连续值）
        description: 数据集描述
    """
    # 设置随机种子，确保结果可重现
    np.random.seed(42)
    
    # 生成1000个样本
    n_samples = 1000
    
    # 特征1：房屋面积（平方米）
    area = np.random.uniform(50, 200, n_samples)
    
    # 特征2：房龄（年）
    age = np.random.uniform(0, 50, n_samples)
    
    # 特征3：房间数量
    rooms = np.random.randint(1, 6, n_samples)
    
    # 特征4：距离市中心的距离（公里）
    distance = np.random.uniform(1, 20, n_samples)
    
    # 组合特征
    X = np.column_stack((area, age, rooms, distance))
    
    # 生成目标值：房价（万元）
    # 基于特征的线性组合加上一些非线性项和噪声
    y = (
        0.8 * area +          # 面积越大，价格越高
        -0.5 * age +          # 房龄越大，价格越低
        15 * rooms +          # 房间越多，价格越高
        -2 * distance +       # 距离越远，价格越低
        0.01 * area**2 +      # 面积的非线性影响
        -0.1 * age * distance +  # 房龄和距离的交互作用
        np.random.normal(0, 10, n_samples)  # 噪声
    )
    
    # 确保价格为正数
    y = np.maximum(y, 30)
    
    description = {
        'name': '回归样本数据集',
        'samples': n_samples,
        'features': X.shape[1],
        'feature_names': ['房屋面积(平方米)', '房龄(年)', '房间数量', '距离市中心(公里)'],
        'target_name': '房价(万元)',
        'description': '该数据集包含房屋相关特征和对应的房价，可用于回归分析。目标是根据房屋特征预测房价。'
    }
    
    return X, y, description
