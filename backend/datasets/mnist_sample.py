import numpy as np

def load_mnist_sample():
    """
    加载MNIST样本数据集（简化版）
    包含0-9数字的手写图像样本，每个图像为28x28像素，这里简化为7x7
    
    返回:
        X: 特征数据，已扁平化处理
        y: 标签数据（0-9）
        description: 数据集描述
    """
    # 为了简化，我们生成一些类似MNIST的样本数据
    # 实际应用中可以加载真实的MNIST数据集
    
    # 设置随机种子，确保结果可重现
    np.random.seed(42)
    
    # 每个数字生成30个样本，共300个样本
    n_samples_per_digit = 30
    digits = 10
    n_samples = n_samples_per_digit * digits
    
    # 图像大小：7x7像素
    img_size = 7
    X = np.zeros((n_samples, img_size * img_size))
    y = np.zeros(n_samples, dtype=int)
    
    # 生成数字0的样本
    for i in range(n_samples_per_digit):
        idx = i
        y[idx] = 0
        # 0的形状：一个圆环
        X[idx, 1*img_size + 1 : 1*img_size + 5] = 1  # 上横
        X[idx, 5*img_size + 1 : 5*img_size + 5] = 1  # 下横
        X[idx, 2*img_size + 1] = 1  # 左竖上
        X[idx, 3*img_size + 1] = 1  # 左竖中
        X[idx, 4*img_size + 1] = 1  # 左竖下
        X[idx, 2*img_size + 5] = 1  # 右竖上
        X[idx, 3*img_size + 5] = 1  # 右竖中
        X[idx, 4*img_size + 5] = 1  # 右竖下
        # 添加一些噪声
        noise = np.random.normal(0, 0.1, img_size * img_size)
        X[idx] = np.clip(X[idx] + noise, 0, 1)
    
    # 生成数字1的样本
    for i in range(n_samples_per_digit):
        idx = n_samples_per_digit + i
        y[idx] = 1
        # 1的形状：一条竖线
        X[idx, 3*img_size + 1 : 3*img_size + 6] = 1  # 竖线
        # 添加一些噪声
        noise = np.random.normal(0, 0.1, img_size * img_size)
        X[idx] = np.clip(X[idx] + noise, 0, 1)
    
    # 生成数字2的样本
    for i in range(n_samples_per_digit):
        idx = 2 * n_samples_per_digit + i
        y[idx] = 2
        # 2的形状
        X[idx, 1*img_size + 1 : 1*img_size + 6] = 1  # 上横
        X[idx, 3*img_size + 3 : 3*img_size + 6] = 1  # 中横右
        X[idx, 5*img_size + 1 : 5*img_size + 6] = 1  # 下横
        X[idx, 2*img_size + 5] = 1  # 右竖上
        X[idx, 4*img_size + 1] = 1  # 左竖下
        # 添加一些噪声
        noise = np.random.normal(0, 0.1, img_size * img_size)
        X[idx] = np.clip(X[idx] + noise, 0, 1)
    
    # 为简化，这里只生成0、1、2三个数字的样本
    # 实际应用中可以生成更多数字
    
    # 截取已生成的样本
    X = X[:3*n_samples_per_digit]
    y = y[:3*n_samples_per_digit]
    
    description = {
        'name': 'MNIST样本数据集',
        'samples': len(X),
        'features': img_size * img_size,
        'classes': len(np.unique(y)),
        'description': 'MNIST数据集包含0-9的手写数字图像，这里是简化版，包含0、1、2三个数字，每个图像为7x7像素，已扁平化处理为49维特征向量。'
    }
    
    return X, y, description
