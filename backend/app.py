import sys
import os

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

# 加载所有数据集
def load_all_datasets():
    """加载所有数据集到内存"""
    datasets['iris'] = load_iris()
    # 后续调用时能正常解包
    X, y, desc = datasets['iris']  # 无报错
    print(desc['description'])     # 能正常打印数据集描述

    datasets['mnist'] = load_mnist_sample()
    datasets['regression'] = load_regression_sample()

# 评估指标计算函数
def calculate_metrics(y_true, y_pred, task_type='classification'):
    """
    计算评估指标
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param task_type: 任务类型，'classification' 或 'regression'
    :return: 评估指标字典
    """
    metrics = {}
    
    if task_type == 'classification':
        # 分类任务指标
        # 计算准确率
        accuracy = np.mean(y_true == y_pred)
        metrics['accuracy'] = accuracy
        
        # 计算精确率、召回率和F1分数（针对二分类）
        classes = np.unique(y_true)
        if len(classes) == 2:
            # 二分类
            positive_class = classes[1]
            
            # 计算TP, TN, FP, FN
            tp = np.sum((y_true == positive_class) & (y_pred == positive_class))
            tn = np.sum((y_true != positive_class) & (y_pred != positive_class))
            fp = np.sum((y_true != positive_class) & (y_pred == positive_class))
            fn = np.sum((y_true == positive_class) & (y_pred != positive_class))
            
            # 精确率
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics['precision'] = precision
            
            # 召回率
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['recall'] = recall
            
            # F1分数
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            metrics['f1'] = f1
    else:
        # 回归任务指标
        # 均方误差
        mse = np.mean((y_true - y_pred) **2)
        metrics['mse'] = mse
        
        # 平均绝对误差
        mae = np.mean(np.abs(y_true - y_pred))
        metrics['mae'] = mae
        
        # 决定系数R²
        ss_total = np.sum((y_true - np.mean(y_true))** 2)
        ss_residual = np.sum((y_true - y_pred) **2)
        r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
        metrics['r2'] = r2
        
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
    
    # 返回部分样本用于可视化
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
    
    # 检查数据集是否存在
    if dataset_id not in datasets or datasets[dataset_id] is None:
        return jsonify({'error': '数据集不存在'}), 404
        
    # 获取数据集
    X, y, _ = datasets[dataset_id]
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 根据算法ID选择并初始化算法
    try:
        if algorithm_id == 'decision_tree':
            model = algos.DecisionTree(max_depth=5)
            task_type = 'classification'
            
        elif algorithm_id == 'naive_bayes':
            model = algos.NaiveBayes()
            task_type = 'classification'
            
        elif algorithm_id == 'knn':
            # 根据数据集类型自动选择KNN的任务类型
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
            # 对于聚类算法，使用所有数据训练，且不需要真实标签
            model = algos.KMeans(k=len(np.unique(y)))
            task_type = 'clustering'
            
        elif algorithm_id == 'em':
            # 对于EM算法，使用所有数据训练
            model = algos.EMAlgorithm(n_components=len(np.unique(y)))
            task_type = 'clustering'
            
        else:
            return jsonify({'error': '算法不存在'}), 404
            
    except Exception as e:
        return jsonify({'error': f'初始化算法失败: {str(e)}'}), 500
        
    # 训练模型
    try:
        if task_type == 'clustering':
            # 聚类算法不需要标签
            model.train(X)
            y_pred = model.predict(X_test)
            # 聚类算法没有真实标签可比较，不计算传统指标
            metrics = {}
        else:
            model.train(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = calculate_metrics(y_test, y_pred, task_type)
            
    except Exception as e:
        return jsonify({'error': f'训练模型失败: {str(e)}'}), 500
        
    # 获取可视化数据
    visualization_data = model.get_visualization_data()
    
    return jsonify({
        'algorithm': algorithm_id,
        'dataset': dataset_id,
        'metrics': metrics,
        'visualization': visualization_data
    })

# API端点：比较多个算法
@app.route('/api/compare', methods=['POST'])
def compare_algorithms():
    data = request.json
    
    if not data or 'algorithms' not in data or 'dataset' not in data or 'metric' not in data:
        return jsonify({'error': '缺少参数'}), 400
        
    algorithm_ids = data['algorithms']
    dataset_id = data['dataset']
    metric = data['metric']
    
    # 检查数据集是否存在
    if dataset_id not in datasets or datasets[dataset_id] is None:
        return jsonify({'error': '数据集不存在'}), 404
        
    # 获取数据集
    X, y, _ = datasets[dataset_id]
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    results = {}
    
    # 逐个训练算法并获取指定指标
    for algorithm_id in algorithm_ids:
        try:
            # 根据算法ID选择并初始化算法
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
                    # 线性回归只适用于回归数据集
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
                
            # 训练模型
            if task_type == 'clustering':
                model.train(X)
                y_pred = model.predict(X_test)
                metrics = {}
            else:
                model.train(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = calculate_metrics(y_test, y_pred, task_type)
                
            # 获取指定指标
            metric_value = metrics.get(metric, None)
            results[algorithm_id] = {'metric': metric_value}
            
        except Exception as e:
            results[algorithm_id] = {'metric': None, 'error': str(e)}
            
    return jsonify({'results': results})

# 启动应用
if __name__ == '__main__':
    # 加载数据集
    load_all_datasets()
    # 启动Flask服务器
    app.run(debug=True)
