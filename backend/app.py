import sys
import os
# 先在文件顶部导入需要的库
from sklearn.metrics import adjusted_rand_score, silhouette_score

# 获取项目根目录（假设 app.py 在 backend 文件夹下，根目录是 backend 的上一级）
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sklearn.model_selection import train_test_split  # 仅用于数据集分割
import algorithms as algos
from datasets.iris import load_iris
from datasets.mnist_sample import load_mnist_sample
from datasets.regression_sample import load_regression_sample


# 初始化Flask应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局存储数据集
datasets = {
    'iris': None,
    'mnist': None,
    'regression': None
}

# NumPy 类型转 Python 类型
def convert_numpy_types(obj):
    """递归将 NumPy 类型转换为 Python 原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # 转换键和值
        new_dict = {}
        for k, v in obj.items():
            new_k = int(k) if isinstance(k, np.integer) else k
            new_dict[new_k] = convert_numpy_types(v)
        return new_dict
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# 加载所有数据集
def load_all_datasets():
    """加载所有数据集到内存"""
    datasets['iris'] = load_iris()
    X, y, desc = datasets['iris']
    print(desc['description'])

    datasets['mnist'] = load_mnist_sample()
    datasets['regression'] = load_regression_sample()

def calculate_metrics(y_true, y_pred, task_type='classification'):
    metrics = {}
    
    # 统一转换为numpy数组，并确保是一维
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # 确保数据类型为浮点数，便于计算
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    
    # 检查形状是否匹配
    if y_true.shape != y_pred.shape:
        raise ValueError(f"真实值和预测值形状不匹配: {y_true.shape} vs {y_pred.shape}")
    
    # 无论分类还是回归都计算MSE
    mse = np.mean((y_true - y_pred) **2)
    metrics['mse'] = float(mse)
    
    if task_type == 'classification':
        # 分类任务特有指标
        accuracy = np.mean(y_true == y_pred)
        metrics['accuracy'] = float(accuracy)
        
        classes = np.unique(y_true)
        precision_list = []
        recall_list = []
        f1_list = []
        
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            precision_list.append(float(precision))
            recall_list.append(float(recall))
            f1_list.append(float(f1))
        
        metrics['precision'] = float(np.mean(precision_list))
        metrics['recall'] = float(np.mean(recall_list))
        metrics['f1'] = float(np.mean(f1_list))
        
    else:  # regression
        # 回归任务特有指标
        rmse = np.sqrt(mse)  # 均方根误差
        metrics['rmse'] = float(rmse)
        
        mae = np.mean(np.abs(y_true - y_pred))  # 平均绝对误差
        metrics['mae'] = float(mae)
        
        # 回归任务不计算分类指标
        metrics['accuracy'] = None
        metrics['precision'] = None
        metrics['recall'] = None
        metrics['f1'] = None
        
    return metrics

# API端点：获取算法列表
@app.route('/api/algorithms', methods=['GET'])
def get_algorithms():
    algorithms = [
        {'id': 'decision_tree', 'name': '决策树', 'task_type': 'classification'},
        {'id': 'naive_bayes', 'name': '朴素贝叶斯', 'task_type': 'classification'},
        {'id': 'knn', 'name': 'K最近邻', 'task_type': 'both'},
        {'id': 'svm', 'name': '支持向量机', 'task_type': 'classification'},
        {'id': 'random_forest', 'name': '随机森林', 'task_type': 'classification'},
        {'id': 'linear_regression', 'name': '线性回归', 'task_type': 'regression'},
        {'id': 'logistic_regression', 'name': '逻辑回归', 'task_type': 'classification'},
        {'id': 'adaboost', 'name': 'AdaBoost', 'task_type': 'classification'},
        {'id': 'kmeans', 'name': 'K均值聚类', 'task_type': 'clustering'},
        {'id': 'em', 'name': 'EM算法', 'task_type': 'clustering'}
    ]
    return jsonify({'algorithms': algorithms})

# API端点：获取数据集列表
@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    datasets_info = [
        {
            'id': 'iris',
            'name': '鸢尾花数据集',
            'task_type': 'classification',
            'description': datasets['iris'][2]['description']
        },
        {
            'id': 'mnist',
            'name': 'MNIST样本',
            'task_type': 'classification',
            'description': datasets['mnist'][2]['description']
        },
        {
            'id': 'regression',
            'name': '回归样本',
            'task_type': 'regression',
            'description': datasets['regression'][2]['description']
        }
    ]
    return jsonify({'datasets': datasets_info})

# API端点：获取特定数据集
@app.route('/api/dataset/<dataset_id>', methods=['GET'])
def get_dataset(dataset_id):
    if dataset_id not in datasets or datasets[dataset_id] is None:
        return jsonify({'error': '数据集不存在'}), 404
        
    X, y, desc = datasets[dataset_id]
    
    sample_size = min(100, len(X))
    indices = np.random.choice(len(X), sample_size, replace=False)
    
    return jsonify({
        'samples': X[indices].tolist(),
        'labels': y[indices].tolist(),
        'description': desc
    })

# API端点：训练模型并返回结果
@app.route('/api/train', methods=['POST'])
def train_model():
    data = request.json
    
    if not data or 'algorithm' not in data or 'dataset' not in data:
        return jsonify({'error': '缺少算法或数据集参数'}), 400
        
    algorithm_id = data['algorithm']
    dataset_id = data['dataset']
    
    if dataset_id not in datasets or datasets[dataset_id] is None:
        return jsonify({'error': '数据集不存在'}), 404
        
    X, y, _ = datasets[dataset_id]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    try:
        if algorithm_id == 'decision_tree':
            model = algos.DecisionTree(max_depth=5)
            task_type = 'classification'
            
        elif algorithm_id == 'naive_bayes':
            model = algos.NaiveBayes()
            task_type = 'classification'
            
        elif algorithm_id == 'knn':
            task_type = 'regression' if dataset_id == 'regression' else 'classification'
            model = algos.KNN(k=5, task_type=task_type)
            
        elif algorithm_id == 'svm':
            model = algos.SVM()
            task_type = 'classification'
            
        elif algorithm_id == 'random_forest':
            model = algos.RandomForest(n_trees=10)
            task_type = 'classification'
            
        elif algorithm_id == 'linear_regression':
            model = algos.LinearRegression()
            task_type = 'regression'
            
        elif algorithm_id == 'logistic_regression':
            model = algos.LogisticRegression()
            task_type = 'classification'
            
        elif algorithm_id == 'adaboost':
            model = algos.AdaBoost(n_estimators=10)
            task_type = 'classification'
            
        elif algorithm_id == 'kmeans':
            model = algos.KMeans(k=len(np.unique(y)))
            task_type = 'clustering'
            
        elif algorithm_id == 'em':
            model = algos.EMAlgorithm(n_components=len(np.unique(y)))
            task_type = 'clustering'
            
        else:
            return jsonify({'error': '算法不存在'}), 404
            
    except Exception as e:
        return jsonify({'error': f'初始化算法失败: {str(e)}'}), 500
        
    try:
        # 找到这部分代码
        if task_type == 'clustering':
            model.train(X)
            y_pred = model.predict(X_test)
            metrics = {}  # 初始化空字典
            
            # 只计算聚类指标，不计算分类指标
            if y is not None and len(y) > 0:
                try:
                    from sklearn.metrics import adjusted_rand_score
                    metrics['ari'] = float(adjusted_rand_score(y_test, y_pred))
                except Exception as e:
                    print(f"计算ARI失败: {e}")
                    metrics['ari'] = None
                    
            try:
                from sklearn.metrics import silhouette_score
                metrics['silhouette'] = float(silhouette_score(X_test, y_pred))
            except Exception as e:
                print(f"计算轮廓系数失败: {e}")
                metrics['silhouette'] = None
        else:
            # 分类/回归算法才计算原来的指标
            model.train(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = calculate_metrics(y_test, y_pred, task_type)
            
    except Exception as e:
        return jsonify({'error': f'训练模型失败: {str(e)}'}), 500
        
    visualization_data = model.get_visualization_data()
    
    response = {
        'algorithm': algorithm_id,
        'dataset': dataset_id,
        'metrics': metrics,
        'visualization': visualization_data
    }
    
    # 转换所有 NumPy 类型
    response = convert_numpy_types(response)
    
    return jsonify(response)

# API端点：比较多个算法
@app.route('/api/compare', methods=['POST'])
def compare_algorithms():
    data = request.json
    
    if not data or 'algorithms' not in data or 'dataset' not in data or 'metric' not in data:
        return jsonify({'error': '缺少参数'}), 400
        
    algorithm_ids = data['algorithms']
    dataset_id = data['dataset']
    metric = data['metric']
    
    if dataset_id not in datasets or datasets[dataset_id] is None:
        return jsonify({'error': '数据集不存在'}), 404
        
    X, y, _ = datasets[dataset_id]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    results = {}
    
    for algorithm_id in algorithm_ids:
        try:
            if algorithm_id == 'decision_tree':
                model = algos.DecisionTree(max_depth=5)
                task_type = 'classification'
                
            elif algorithm_id == 'naive_bayes':
                model = algos.NaiveBayes()
                task_type = 'classification'
                
            elif algorithm_id == 'knn':
                task_type = 'regression' if dataset_id == 'regression' else 'classification'
                model = algos.KNN(k=5, task_type=task_type)
                
            elif algorithm_id == 'svm':
                model = algos.SVM()
                task_type = 'classification'
                
            elif algorithm_id == 'random_forest':
                model = algos.RandomForest(n_trees=10)
                task_type = 'classification'
                
            elif algorithm_id == 'linear_regression':
                if dataset_id != 'regression':
                    results[algorithm_id] = {'metric': None, 'error': '不适用于分类任务'}
                    continue
                model = algos.LinearRegression()
                task_type = 'regression'
                
            elif algorithm_id == 'logistic_regression':
                model = algos.LogisticRegression()
                task_type = 'classification'
                
            elif algorithm_id == 'adaboost':
                model = algos.AdaBoost(n_estimators=10)
                task_type = 'classification'
                
            elif algorithm_id == 'kmeans':
                model = algos.KMeans(k=len(np.unique(y)))
                task_type = 'clustering'
                
            elif algorithm_id == 'em':
                model = algos.EMAlgorithm(n_components=len(np.unique(y)))
                task_type = 'clustering'
                
            else:
                results[algorithm_id] = {'metric': None, 'error': '算法不存在'}
                continue
                
            if task_type == 'clustering':
                model.train(X)
                y_pred = model.predict(X_test)
                metrics = {}

                            # 如果有真实标签，可以计算聚类评估指标
                if y is not None and len(y) > 0:
                    try:
                        # 计算调整兰德指数(ARI)
                        metrics['ari'] = float(adjusted_rand_score(y_test, y_pred))
                    except Exception:
                        metrics['ari'] = None
                        
                # 计算轮廓系数(Silhouette Score)
                try:
                    metrics['silhouette'] = float(silhouette_score(X_test, y_pred))
                except Exception:
                    metrics['silhouette'] = None    
            else:
                model.train(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = calculate_metrics(y_test, y_pred, task_type)
                
            metric_value = metrics.get(metric, None)
            results[algorithm_id] = {'metric': metric_value}
            
        except Exception as e:
            results[algorithm_id] = {'metric': None, 'error': str(e)}
    
    # 转换所有 NumPy 类型
    results = convert_numpy_types(results)
            
    return jsonify({'results': results})

# 启动应用
if __name__ == '__main__':
    load_all_datasets()
    app.run(debug=True)