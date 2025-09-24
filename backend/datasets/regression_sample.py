import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

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
    # 调整特征影响权重，使模式更明显
    y = (
        1.2 * area +          # 面积越大，价格越高（增加权重）
        -1.0 * age +          # 房龄越大，价格越低（增加权重）
        20 * rooms +          # 房间越多，价格越高（增加权重）
        -3 * distance +       # 距离越远，价格越低（增加权重）
        0.01 * area**2 +      # 面积的非线性影响
        -0.15 * age * distance +  # 房龄和距离的交互作用（增加权重）
        np.random.normal(0, 8, n_samples)  # 减少噪声
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

def evaluate_dataset():
    # 加载数据
    X, y, desc = load_regression_sample()
    print(f"数据集信息: {desc['name']}, 样本数: {desc['samples']}, 特征数: {desc['features']}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 尝试不同的模型
    models = {
        "线性回归": LinearRegression(),
        "岭回归": Ridge(alpha=1.0),
        "随机森林回归": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    # 评估每个模型
    for name, model in models.items():
        # 训练模型
        model.fit(X_train_scaled, y_train)
        
        # 预测
        y_pred = model.predict(X_test_scaled)
        
        # 评估
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n{name} 结果:")
        print(f"均方根误差 (RMSE): {rmse:.2f}")
        print(f"R² 分数: {r2:.4f} (越接近1越好)")
