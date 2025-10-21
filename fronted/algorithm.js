// 后端API地址
const API_BASE_URL = 'http://localhost:5000/api';

// 算法描述数据
const algorithmDescriptions = {
    decision_tree: {
        title: "决策树算法",
        description: "决策树是一种树形结构，其中每个内部节点表示一个属性上的判断，每个分支代表一个判断结果的输出，每个叶节点代表一种分类结果。决策树易于理解和解释，可以处理数值型和类别型数据，不需要预处理数据。适用于分类和回归任务，但容易过拟合。"
    },
    naive_bayes: {
        title: "朴素贝叶斯算法",
        description: "朴素贝叶斯算法基于贝叶斯定理和特征条件独立性假设。它通过计算后验概率来进行分类，计算简单且高效。尽管'朴素'地假设特征之间相互独立，但在实际应用中表现良好，特别适用于文本分类、垃圾邮件过滤等场景。"
    },
    knn: {
        title: "K最近邻算法 (KNN)",
        description: "KNN是一种基于实例的学习算法，它通过计算新样本与训练集中所有样本的距离，选取最近的K个样本，根据这K个样本的多数类别（分类）或平均值（回归）来预测新样本的类别或值。KNN简单直观，但计算复杂度高，对异常值和噪声敏感。"
    },
    svm: {
        title: "支持向量机 (SVM)",
        description: "SVM通过寻找一个最优超平面来实现分类，这个超平面能最大化两类样本之间的间隔。对于非线性问题，SVM使用核函数将数据映射到高维空间，从而可以在高维空间中找到线性分离超平面。SVM在小样本、高维空间中表现良好，但计算复杂度高，对参数和核函数选择敏感。"
    },
    random_forest: {
        title: "随机森林",
        description: "随机森林是一种集成学习方法，它构建多个决策树，并通过投票（分类）或平均（回归）来确定最终结果。通过引入随机性（随机选择样本和特征），随机森林可以有效避免过拟合，提高模型的泛化能力。随机森林性能优异，鲁棒性强，但模型解释性较差。"
    },
    linear_regression: {
        title: "线性回归",
        description: "线性回归用于建立自变量与因变量之间的线性关系模型，通过最小化预测值与实际值之间的平方误差来求解模型参数。线性回归简单易解释，但只能捕捉线性关系，对非线性数据拟合效果差。适用于预测连续值的回归任务。"
    },
    logistic_regression: {
        title: "逻辑回归",
        description: "逻辑回归虽然名为回归，但实际上是一种分类算法。它通过Sigmoid函数将线性回归的输出映射到[0,1]区间，得到样本属于某一类别的概率。逻辑回归实现简单，解释性强，适用于二分类问题，但表达能力有限，难以处理复杂的非线性关系。"
    },
    adaboost: {
        title: "AdaBoost算法",
        description: "AdaBoost是一种集成学习方法，它通过迭代的方式训练多个弱分类器，并根据每个分类器的性能分配不同的权重，最后将这些弱分类器加权组合成一个强分类器。AdaBoost对噪声敏感，但在许多实际问题上表现出色，实现简单且不易过拟合。"
    },
    kmeans: {
        title: "K均值聚类",
        description: "K均值聚类是一种无监督学习算法，它将数据集划分为K个簇，使得同一簇内的数据点相似度高，不同簇的数据点相似度低。算法通过迭代更新簇中心来优化聚类结果。K均值聚类简单高效，但需要预先指定K值，对初始簇中心敏感，且对非凸形状的簇聚类效果较差。"
    },
    em: {
        title: "EM算法 (期望最大化算法)",
        description: "EM算法是一种用于含有隐变量模型的参数估计方法，由期望步（E步）和最大化步（M步）交替组成。E步计算隐变量的后验概率，M步基于E步的结果更新模型参数。EM算法广泛应用于混合模型、因子分析等场景，但容易陷入局部最优，收敛速度可能较慢。"
    }
};

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    // 初始化算法和数据集选择器
    initSelectors();
    
    // 初始化算法概览卡片
    initAlgorithmGrid();
    
    // 初始化算法选择事件
    initAlgorithmSelection();
    
    // 初始化数据集查看事件
    initDatasetViewer();
    
    // 初始化比较图表
    initComparisonChart();
    
    // 初始化报告生成功能
    initReportGeneration();
    
    // 初始化画布上下文
    initCanvases();
});

// 初始化算法和数据集选择器
async function initSelectors() {
    try {
        // 获取算法列表
        const algorithmsResponse = await fetch(`${API_BASE_URL}/algorithms`);
        const algorithmsData = await algorithmsResponse.json();
        
        // 填充算法选择器
        const algorithmSelect = document.getElementById('algorithmSelect');
        algorithmsData.algorithms.forEach(algo => {
            const option = document.createElement('option');
            option.value = algo.id;
            option.textContent = algo.name;
            algorithmSelect.appendChild(option);
        });

        // 获取数据集列表
        const datasetsResponse = await fetch(`${API_BASE_URL}/datasets`);
        const datasetsData = await datasetsResponse.json();
        
        // 填充数据集选择器
        const datasetSelect = document.getElementById('datasetSelect');
        datasetsData.datasets.forEach(dataset => {
            const option = document.createElement('option');
            option.value = dataset.id;
            option.textContent = dataset.name;
            datasetSelect.appendChild(option);
        });

        // 2. 修正：获取数据集划分比例（原错误：复用了algorithmsResponse，且ID/Name错误）
        const divisionResponse = await fetch(`${API_BASE_URL}/divisions`);
        console.log('划分比例接口响应状态:', divisionResponse.status); // 新增日志
        
        const divisionData = await divisionResponse.json();
        console.log('获取到的划分比例数据:', divisionData); // 新增日志
        
        const divisionSelect = document.getElementById('divisionSelect');
        console.log('是否找到划分比例选择器:', divisionSelect !== null); // 新增日志
        
        // 清空并填充选项
        divisionSelect.innerHTML = '';
        divisionData.divisions.forEach(division => {
            const option = document.createElement('option');
            option.value = division.id;
            option.textContent = division.name;
            divisionSelect.appendChild(option);
            console.log('添加选项:', division.id, division.test_size); // 新增日志
        });
        
        // 设置默认选中项
        divisionSelect.value = '7:3';
        console.log('默认选中的划分比例ID:', divisionSelect.value); // 新增日志
        
    } catch (error) {
        console.error('初始化选择器失败:', error);
        alert('无法加载划分比例，请检查后端服务');
    }
}

// 初始化算法概览卡片
function initAlgorithmGrid() {
    const grid = document.getElementById('algorithmGrid');
    
    // 清空网格
    grid.innerHTML = '';
    
    // 为每个算法创建卡片
    for (const [key, algo] of Object.entries(algorithmDescriptions)) {
        const card = document.createElement('div');
        card.className = 'algorithm-card';
        card.dataset.algorithm = key;
        
        card.innerHTML = `
            <h3>${algo.title}</h3>
            <p>${algo.description.substring(0, 100)}...</p>
        `;
        
        // 添加点击事件
        card.addEventListener('click', function() {
            document.getElementById('algorithmSelect').value = key;
            showAlgorithmDetails(key);
            // 滚动到算法详情部分
            document.getElementById('algorithms').scrollIntoView({ behavior: 'smooth' });
        });
        
        grid.appendChild(card);
    }
}

// 初始化算法选择事件
function initAlgorithmSelection() {
    const algorithmSelect = document.getElementById('algorithmSelect');
    const runButton = document.getElementById('runAlgorithm');
    
    algorithmSelect.addEventListener('change', function() {
        const algoKey = this.value;
        if (algoKey) {
            showAlgorithmDetails(algoKey);
        }
    });
    
    runButton.addEventListener('click', async function() {
        const algoKey = document.getElementById('algorithmSelect').value;
        const datasetKey = document.getElementById('datasetSelect').value;
        
        if (algoKey && datasetKey) {
            await runAlgorithm(algoKey, datasetKey);
        } else {
            alert('请先选择算法和数据集');
        }
    });
}

// 显示算法详情
function showAlgorithmDetails(algoKey) {
    const algo = algorithmDescriptions[algoKey];
    if (algo) {
        document.getElementById('algorithmTitle').textContent = algo.title;
        document.getElementById('algorithmDescription').textContent = algo.description;
        
        // 绘制算法示意图
        drawAlgorithmDiagram(algoKey);
    }
}

// 绘制算法示意图
function drawAlgorithmDiagram(algoKey) {
    const canvas = document.getElementById('algorithmVisualization');
    const ctx = canvas.getContext('2d');
    
    // 清除画布
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // 设置画布尺寸
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    
    // 根据不同算法绘制不同的示意图
    switch (algoKey) {
        case 'decision_tree':
            drawDecisionTree(ctx, canvas.width, canvas.height);
            break;
        case 'naive_bayes':
            drawNaiveBayes(ctx, canvas.width, canvas.height);
            break;
        case 'knn':
            drawKNN(ctx, canvas.width, canvas.height);
            break;
        case 'svm':
            drawSVM(ctx, canvas.width, canvas.height);
            break;
        case 'random_forest':
            drawRandomForest(ctx, canvas.width, canvas.height);
            break;
        case 'linear_regression':
            drawLinearRegression(ctx, canvas.width, canvas.height);
            break;
        case 'logistic_regression':
            drawLogisticRegression(ctx, canvas.width, canvas.height);
            break;
        case 'adaboost':
            drawAdaBoost(ctx, canvas.width, canvas.height);
            break;
        case 'kmeans':
            drawKMeans(ctx, canvas.width, canvas.height);
            break;
        case 'em':
            drawEM(ctx, canvas.width, canvas.height);
            break;
    }
}

// 初始化数据集查看事件
function initDatasetViewer() {
    const datasetButtons = document.querySelectorAll('.view-dataset');
    
    datasetButtons.forEach(button => {
        button.addEventListener('click', async function() {
            const datasetKey = this.closest('.dataset-card').dataset.dataset;
            await visualizeDataset(datasetKey);
        });
    });
}

async function visualizeDataset(datasetKey) {
    try {
        const response = await fetch(`${API_BASE_URL}/dataset/${datasetKey}`);
        
        if (!response.ok) {
            throw new Error(`HTTP错误！状态: ${response.status}`);
        }
        
        const dataset = await response.json();
        
        const container = document.getElementById('datasetVisualization');
        container.innerHTML = '';
        
        const canvas = document.createElement('canvas');
        canvas.id = 'datasetCanvas';
        
        // 关键：设置最小尺寸或根据容器尺寸设置
        const containerWidth = container.offsetWidth || 800;
        const containerHeight = container.offsetHeight || 600;
        canvas.width = containerWidth;
        canvas.height = containerHeight;
        
        container.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        
        // 根据数据集类型调用相应的绘制函数
        switch (datasetKey) {
            case 'iris':
                drawIrisDataset(ctx, canvas.width, canvas.height, dataset);
                break;
            case 'mnist':
                drawMNISTDataset(ctx, canvas.width, canvas.height, dataset);
                break;
            case 'regression':
                drawRegressionDataset(ctx, canvas.width, canvas.height, dataset);
                break;
        }
        
        // 可选：窗口大小变化时重新绘制
        function handleResize() {
            const w = container.offsetWidth || 800;
            const h = container.offsetHeight || 600;
            canvas.width = w;
            canvas.height = h;
            
            // 重新绘制
            switch (datasetKey) {
                case 'iris':
                    drawIrisDataset(ctx, w, h, dataset);
                    break;
                case 'mnist':
                    drawMNISTDataset(ctx, w, h, dataset);
                    break;
                case 'regression':
                    drawRegressionDataset(ctx, w, h, dataset);
                    break;
            }
        }
        
        window.addEventListener('resize', handleResize);
        // 保存清理函数，避免重复绑定
        canvas._onResize = handleResize;
        
    } catch (error) {
        console.error('可视化数据集失败:', error);
        alert('获取数据集失败，请重试。');
    }
}

// 初始化比较图表
function initComparisonChart() {
    const updateButton = document.getElementById('updateComparison');
    
    updateButton.addEventListener('click', async function() {
        const dataset = document.getElementById('comparisonDataset').value;
        const metric = document.getElementById('comparisonMetric').value;
        await updateComparisonChart(dataset, metric);
    });
    
    // 初始加载
    updateComparisonChart('iris', 'accuracy');
}

// 更新比较图表
async function updateComparisonChart(dataset, metric) {
    try {
        // 获取所有算法
        const algorithmsResponse = await fetch(`${API_BASE_URL}/algorithms`);
        const algorithmsData = await algorithmsResponse.json();
        const algorithmIds = algorithmsData.algorithms.map(algo => algo.id);
        
        // 请求比较结果
        const response = await fetch(`${API_BASE_URL}/compare`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                algorithms: algorithmIds,
                dataset: dataset,
                metric: metric
            })
        });
        
        const comparisonData = await response.json();
        
        const canvas = document.getElementById('comparisonChart');
        const ctx = canvas.getContext('2d');
        
        // 设置画布尺寸
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
        
        // 清除画布
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // 绘制比较图表
        const results = comparisonData.results;
        
        // 过滤掉值为null的数据
        const filteredData = {};
        for (const [algo, data] of Object.entries(results)) {
            if (data.metric !== null && !isNaN(data.metric)) {
                filteredData[algo] = data;
            }
        }
        
        // 绘制柱状图
        const algorithms = Object.keys(filteredData);
        const values = algorithms.map(algo => filteredData[algo].metric);
        const numBars = algorithms.length;
        
        if (numBars === 0) {
            ctx.fillStyle = '#333';
            ctx.font = '16px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('所选指标不适用于该数据集的任何算法', canvas.width / 2, canvas.height / 2);
            return;
        }
        
        const barWidth = canvas.width * 0.6 / numBars;
        const barSpacing = canvas.width * 0.3 / (numBars + 1);
        const startX = canvas.width * 0.15 + barSpacing;
        const maxValue = Math.max(...values) * 1.1;
        const barHeightScale = canvas.height * 0.7 / maxValue;
        const startY = canvas.height * 0.85;
        
        // 绘制坐标轴
        ctx.beginPath();
        ctx.moveTo(canvas.width * 0.1, startY);
        ctx.lineTo(canvas.width * 0.9, startY);
        ctx.lineTo(canvas.width * 0.88, startY - 5);
        ctx.moveTo(canvas.width * 0.9, startY);
        ctx.lineTo(canvas.width * 0.88, startY + 5);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(canvas.width * 0.1, startY);
        ctx.lineTo(canvas.width * 0.1, canvas.height * 0.15);
        ctx.lineTo(canvas.width * 0.08, canvas.height * 0.17);
        ctx.moveTo(canvas.width * 0.1, canvas.height * 0.15);
        ctx.lineTo(canvas.width * 0.12, canvas.height * 0.17);
        ctx.stroke();
        
        // 绘制刻度和标签
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        for (let i = 0; i <= 5; i++) {
            const y = startY - (i * maxValue / 5) * barHeightScale;
            ctx.beginPath();
            ctx.moveTo(canvas.width * 0.1 - 5, y);
            ctx.lineTo(canvas.width * 0.1, y);
            ctx.stroke();
            
            ctx.fillText((i * maxValue / 5).toFixed(2), canvas.width * 0.08, y + 4);
        }
        
        // 绘制柱状图
        algorithms.forEach((algo, index) => {
            const value = filteredData[algo].metric;
            const barHeight = value * barHeightScale;
            const x = startX + index * (barWidth + barSpacing);
            const y = startY - barHeight;
            
            // 随机颜色
            const color = `hsl(${index * 360 / numBars}, 70%, 60%)`;
            
            // 绘制柱子
            ctx.fillStyle = color;
            ctx.fillRect(x, y, barWidth, barHeight);
            
            // 绘制柱子边框
            ctx.strokeStyle = '#333';
            ctx.strokeRect(x, y, barWidth, barHeight);
            
            // 绘制值标签
            ctx.fillStyle = '#333';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(value.toFixed(4), x + barWidth / 2, y - 10);
            
            // 绘制算法名称标签
            ctx.fillText(algorithmDescriptions[algo]?.title.substring(0, 6) + '...', 
                        x + barWidth / 2, startY + 20);
        });
        
        // 绘制图表标题
        ctx.font = '16px Arial bold';
        ctx.textAlign = 'center';
        ctx.fillText(`${metricToLabel(metric)}比较 (${datasetToLabel(dataset)})`, 
                    canvas.width / 2, canvas.height * 0.1);
    } catch (error) {
        console.error('更新比较图表失败:', error);
        alert('更新比较图表失败，请重试。');
    }
}

// 指标名称转换
function metricToLabel(metric) {
    const labels = {
        accuracy: '准确率',
        precision: '精确率',
        recall: '召回率',
        f1: 'F1分数',
        mse: '均方误差'
    };
    return labels[metric] || metric;
}

// 数据集名称转换
function datasetToLabel(dataset) {
    const labels = {
        iris: '鸢尾花数据集',
        mnist: 'MNIST样本',
        regression: '回归样本'
    };
    return labels[dataset] || dataset;
}

// 初始化报告生成功能
function initReportGeneration() {
    const generateButton = document.getElementById('generateReport');
    
    generateButton.addEventListener('click', async function() {
        await generateExperimentReport();
    });
}

// 生成实验报告
async function generateExperimentReport() {
    try {
        const algorithmsResponse = await fetch(`${API_BASE_URL}/algorithms`);
        const algorithmsData = await algorithmsResponse.json();
        
        // 按任务类型分组算法
        const clusteringAlgos = algorithmsData.algorithms
            .filter(algo => algo.task_type === 'clustering')
            .map(algo => algo.id);
            
        const classificationAlgos = algorithmsData.algorithms
            .filter(algo => algo.task_type === 'classification' || algo.task_type === 'both')
            .map(algo => algo.id);
            
        const regressionAlgos = algorithmsData.algorithms
            .filter(algo => algo.task_type === 'regression' || algo.task_type === 'both')
            .map(algo => algo.id);
        
        const comparisonResults = {};
        
        // 1. 鸢尾花数据集（分类）
        comparisonResults.iris = await fetch(`${API_BASE_URL}/compare`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                algorithms: classificationAlgos,
                dataset: 'iris',
                metric: 'accuracy'
            })
        }).then(r => r.json());
        
        // 2. MNIST数据集（分类）
        comparisonResults.mnist = await fetch(`${API_BASE_URL}/compare`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                algorithms: classificationAlgos,
                dataset: 'mnist',
                metric: 'accuracy'
            })
        }).then(r => r.json());
        
        // 3. 回归数据集（回归）
        comparisonResults.regression = await fetch(`${API_BASE_URL}/compare`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                algorithms: regressionAlgos,
                dataset: 'regression',
                metric: 'mse'
            })
        }).then(r => r.json());
        
        let reportContent = `<h3>机器学习算法性能比较实验报告</h3>`;
        
        // 鸢尾花数据集（全部算法）
        const irisResults = comparisonResults.iris.results;
        const allIris = Object.entries(irisResults)
            .filter(([_, data]) => typeof data.metric === 'number')
            .sort(([_, a], [__, b]) => b.metric - a.metric);  // 准确率高到低
            
        reportContent += `<h4>1. 鸢尾花数据集（分类任务）</h4>`;
        reportContent += `<ul>`;
        allIris.forEach(([algoId, data], index) => {
            reportContent += `<li>${index + 1}. ${algorithmDescriptions[algoId]?.title || algoId}：准确率 ${data.metric.toFixed(4)}</li>`;
        });
        reportContent += `</ul>`;
        
        // MNIST数据集（全部算法）
        const mnistResults = comparisonResults.mnist.results;
        const allMnist = Object.entries(mnistResults)
            .filter(([_, data]) => typeof data.metric === 'number')
            .sort(([_, a], [__, b]) => b.metric - a.metric);  // 准确率高到低
            
        reportContent += `<h4>2. MNIST样本数据集（分类任务）</h4>`;
        reportContent += `<ul>`;
        allMnist.forEach(([algoId, data], index) => {
            reportContent += `<li>${index + 1}. ${algorithmDescriptions[algoId]?.title || algoId}：准确率 ${data.metric.toFixed(4)}</li>`;
        });
        reportContent += `</ul>`;
        
        // 回归数据集（全部算法）
        const regressionResults = comparisonResults.regression.results;
        const allRegression = Object.entries(regressionResults)
            .filter(([_, data]) => typeof data.metric === 'number')
            .sort(([_, a], [__, b]) => a.metric - b.metric);  // MSE低到高
            
        reportContent += `<h4>3. 回归样本数据集（回归任务）</h4>`;
        reportContent += `<ul>`;
        allRegression.forEach(([algoId, data], index) => {
            reportContent += `<li>${index + 1}. ${algorithmDescriptions[algoId]?.title || algoId}：均方误差 ${data.metric.toFixed(4)}</li>`;
        });
        reportContent += `</ul>`;
        
        // 总结
        reportContent += `<h4>4. 总结</h4>`;
        reportContent += `<p>实验结果表明，不同的机器学习算法在不同类型的任务和数据集上表现各有优劣。</p>`;
        reportContent += `<p>对于分类任务，SVM和随机森林通常表现较好；对于回归任务，随机森林和线性回归是不错的选择。</p>`;
        reportContent += `<p>在实际应用中，应根据具体问题类型、数据集特点和计算资源等因素选择合适的算法。</p>`;
        
        document.getElementById('experimentResults').innerHTML = reportContent;
        
    } catch (error) {
        console.error('生成实验报告失败:', error);
        alert('生成实验报告失败，请重试。');
    }
}


// 初始化画布上下文
function initCanvases() {
    const canvases = document.querySelectorAll('canvas');
    canvases.forEach(canvas => {
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
    });
    
    // 窗口大小改变时重新设置画布尺寸
    window.addEventListener('resize', function() {
        canvases.forEach(canvas => {
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
        });
        
        // 重新绘制当前显示的图表
        const selectedAlgo = document.getElementById('algorithmSelect').value;
        if (selectedAlgo) {
            drawAlgorithmDiagram(selectedAlgo);
        }
        
        const dataset = document.getElementById('comparisonDataset').value;
        const metric = document.getElementById('comparisonMetric').value;
        updateComparisonChart(dataset, metric);
    });
}

async function runAlgorithm(algoKey, datasetKey) {
    try {
        // 显示加载状态
        document.getElementById('accuracy').textContent = '加载中...';
        document.getElementById('precision').textContent = '加载中...';
        document.getElementById('recall').textContent = '加载中...';
        document.getElementById('f1').textContent = '加载中...';
        document.getElementById('mse').textContent = '加载中...';
        
        // 1. 获取划分比例选择器
        const divisionSelect = document.getElementById('divisionSelect');
        if (!divisionSelect) {
            alert('未找到划分比例选择器，请检查HTML');
            return;
        }
        
        // 2. 获取用户选择的划分比例ID（如"7:3"）
        const selectedDivisionId = divisionSelect.value;
        if (!selectedDivisionId) {
            alert('请选择数据集划分比例');
            return;
        }
        
        // 3. 请求所有划分比例选项，找到对应的test_size
        const divisionResponse = await fetch(`${API_BASE_URL}/divisions`);
        const divisionData = await divisionResponse.json();
        const selectedDivision = divisionData.divisions.find(
            div => div.id === selectedDivisionId
        );
        
        if (!selectedDivision) {
            alert('未找到选中的划分比例，请重试');
            return;
        }
        const test_size = selectedDivision.test_size; // 提取test_size
        // ------------------------------------------------------
        
        console.log(`开始运行算法: ${algoKey}, 数据集: ${datasetKey}, 测试集比例: ${test_size}`);
        
        // 发送请求时，在body中添加test_size
        const response = await fetch(`${API_BASE_URL}/train`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                algorithm: algoKey,
                dataset: datasetKey,
                test_size: test_size // 关键：传递test_size
            })
        });
        
        if (!response.ok) throw new Error(`HTTP错误！状态: ${response.status}`);
        
        const result = await response.json();
        console.log('服务器返回数据:', result);
        
        if (result.error) {
            console.error(`业务错误: ${result.error}`);
            alert(`错误: ${result.error}`);
            return;
        }
        
        if (!result.metrics) {
            console.error('服务器返回数据缺少metrics字段', result);
            alert('算法运行结果格式不正确');
            return;
        }
        
        if (result.visualization) {
            window.visualizationData = result.visualization;
            if (result.visualization.data) {
                window.currentDatasetFeatures = result.visualization.data;
            }
        }
        
        function formatNumber(value) {
            if (value === null || value === undefined || typeof value !== 'number' || isNaN(value)) {
                return '--';
            }
            return value.toFixed(4);
        }

        // 替换原来的指标显示部分
        if (algoKey === 'kmeans' || algoKey === 'em') {
            // 聚类算法：只显示ARI和轮廓系数
            document.getElementById('accuracy').textContent = `ARI(兰德指数):${formatNumber(result.metrics.ari || '--')}`;
            document.getElementById('precision').textContent = `轮廓系数:${formatNumber(result.metrics.silhouette || '--')}`;
            document.getElementById('recall').textContent = '不适用';
            document.getElementById('f1').textContent = '不适用';
            document.getElementById('mse').textContent = '不适用';
        } else {
            // 其他算法：正常显示分类/回归指标
            document.getElementById('accuracy').textContent = formatNumber(result.metrics.accuracy);
            document.getElementById('precision').textContent = formatNumber(result.metrics.precision);
            document.getElementById('recall').textContent = formatNumber(result.metrics.recall);
            document.getElementById('f1').textContent = formatNumber(result.metrics.f1);
            document.getElementById('mse').textContent = formatNumber(result.metrics.mse);
        }
        // =========== 修改结束 ============
        
        drawAlgorithmResults(algoKey, datasetKey, result.metrics, result.visualization);
        console.log(`算法${algoKey}运行成功`);
        
    } catch (error) {
        console.error('运行算法时发生异常:', error);
        alert(`运行算法失败: ${error.message}\n请查看控制台获取详细信息`);
    }
}


// 绘制算法结果可视化
function drawAlgorithmResults(algoKey, datasetKey, metrics, visualizationData) {
    // 首先绘制算法基本示意图
    drawAlgorithmDiagram(algoKey);
    
    const canvas = document.getElementById('algorithmVisualization');
    const ctx = canvas.getContext('2d');
    
    // 在图表上方绘制性能指标
    ctx.fillStyle = '#333';
    ctx.font = '14px Arial bold';
    ctx.textAlign = 'center';
    ctx.fillText('算法性能结果', canvas.width / 2, 30);
    
    // 绘制性能指标卡片
    const metricsList = [];
    if (metrics.accuracy !== undefined) metricsList.push(`准确率: ${metrics.accuracy.toFixed(4)}`);
    if (metrics.precision !== undefined) metricsList.push(`精确率: ${metrics.precision.toFixed(4)}`);
    if (metrics.recall !== undefined) metricsList.push(`召回率: ${metrics.recall.toFixed(4)}`);
    if (metrics.f1 !== undefined) metricsList.push(`F1分数: ${metrics.f1.toFixed(4)}`);
    if (metrics.mse !== undefined) metricsList.push(`均方误差: ${metrics.mse.toFixed(4)}`);
    
    metricsList.forEach((metric, index) => {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.fillRect(canvas.width - 200, 50 + index * 30, 180, 25);
        
        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(metric, canvas.width - 190, 68 + index * 30);
    });
    
    // 如果有算法特定的可视化数据，在这里处理
    if (visualizationData) {
        // 例如，对于决策树，可以绘制树结构
        if (algoKey === 'decision_tree') {
            // drawTreeVisualization(ctx, canvas.width, canvas.height);
            drawDecisionTree(ctx, canvas.width, canvas.height);
        }
        else if (algoKey === 'kmeans') {
            drawKMeansResults(ctx, canvas.width, canvas.height);
        }
        else if (algoKey === 'knn') {
            drawKNN(ctx, canvas.width, canvas.height);
        }
        else if (algoKey === 'adaboost') {
            drawAdaBoost(ctx, canvas.width, canvas.height);
        }
        else if (algoKey === 'em') {
            drawEM(ctx, canvas.width, canvas.height);
        }
        else if (algoKey === 'linear_regression') {
            drawLinearRegression(ctx, canvas.width, canvas.height);
        }
        else if (algoKey === 'logistic_regression') {
            drawLogisticRegression(ctx, canvas.width, canvas.height);
        }
        else if (algoKey === 'random_forest') {
            drawRandomForest(ctx, canvas.width, canvas.height);
        }
        else if (algoKey === 'naive_bayes') {
            drawNaiveBayes(ctx, canvas.width, canvas.height);
        }
        else if (algoKey === 'svm') {
            drawSVM(ctx, canvas.width, canvas.height);
        }
    }
}

// 以下是各种算法的可视化绘制函数和数据集可视化函数
// 这些函数与前面提供的版本类似，但增加了对实际数据的处理
// 为简洁起见，这里省略具体实现，实际使用时需要完整实现

// 绘制决策树
// 绘制决策树
function drawDecisionTree(ctx, width, height) {
    // 绘制根节点
    drawNode(ctx, width / 2, 50, 100, 40, "特征A > 5?");
    
    // 绘制左子树
    drawLine(ctx, width / 2, 70, width / 3, 120);
    drawNode(ctx, width / 3, 150, 80, 40, "是");
    drawLine(ctx, width / 3, 170, width / 4, 220);
    drawNode(ctx, width / 4, 250, 80, 40, "类别1");
    
    // 绘制右子树
    drawLine(ctx, width / 2, 70, 2 * width / 3, 120);
    drawNode(ctx, 2 * width / 3, 150, 80, 40, "否");
    drawLine(ctx, 2 * width / 3, 170, 3 * width / 4, 220);
    drawNode(ctx, 3 * width / 4, 250, 80, 40, "类别2");
}

// 绘制朴素贝叶斯
function drawNaiveBayes(ctx, width, height) {
    // 绘制贝叶斯公式
    ctx.fillStyle = '#333';
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText("P(A|B) = P(B|A) * P(A) / P(B)", width / 2, height / 3);
    
    // 绘制文本说明
    ctx.font = '14px Arial';
    ctx.fillText("后验概率 = 似然度 * 先验概率 / 证据", width / 2, height / 3 + 30);
    
    // 绘制特征独立性示意图
    const features = ["特征1", "特征2", "特征3", "特征4"];
    const centerX = width / 2;
    const centerY = height * 2 / 3;
    const radius = 80;
    
    // 绘制中心节点（类别）
    drawNode(ctx, centerX, centerY, 80, 40, "类别");
    
    // 绘制特征节点
    features.forEach((feature, i) => {
        const angle = (i / features.length) * Math.PI * 2;
        const x = centerX + Math.cos(angle) * radius;
        const y = centerY + Math.sin(angle) * radius;
        
        drawNode(ctx, x, y, 70, 35, feature);
        drawLine(ctx, centerX, centerY, x, y);
    });
}


function drawKMeansResults(ctx, width, height) {
    // 1. 检查上下文和尺寸
    if (!ctx) {
        console.error('绘图上下文(ctx)为空');
        alert('可视化失败：绘图上下文无效');
        return;
    }
    if (!width || !height || width <= 0 || height <= 0) {
        console.error('无效的画布尺寸', {width, height});
        alert('可视化失败：画布尺寸无效');
        return;
    }

    // 2. 从全局获取数据
    if (!window.visualizationData) {
        console.error('全局变量 window.visualizationData 不存在');
        alert('可视化失败：未提供数据');
         return;
    }
    const { centroids, labels, k: kFromData } = window.visualizationData;

    // 3. 检查数据结构
    if (!centroids || !Array.isArray(centroids)) {
        console.error('centroids数据无效', centroids);
        alert('可视化失败：聚类中心数据无效');
        return;
    }
    if (!labels || !Array.isArray(labels)) {
        console.error('labels数据无效', labels);
        alert('可视化失败：标签数据无效');
        return;
    }

    // 4. 清空画布
    ctx.clearRect(0, 0, width, height);

    // 5. 准备数据
    const k = kFromData || centroids.length;

    // 6. 检查原始数据
    if (!window.currentDatasetFeatures || !Array.isArray(window.currentDatasetFeatures)) {
        console.error('缺少原始数据 window.currentDatasetFeatures');
        alert('可视化失败：缺少原始数据');
        return;
    }

    // 7. 处理数据长度
    const dataLen = Math.min(window.currentDatasetFeatures.length, labels.length);
    if (dataLen === 0) {
        console.error('没有可可视化的数据');
        alert('可视化失败：没有可显示的数据');
        return;
    }

    // 8. 数据归一化
    const features = window.currentDatasetFeatures.slice(0, dataLen);
    const feature0 = features.map(s => s[0]);
    const feature1 = features.map(s => s[1]);

    const minX = Math.min(...feature0);
    const maxX = Math.max(...feature0);
    const minY = Math.min(...feature1);
    const maxY = Math.max(...feature1);

    const normalizeX = (x) => {
        const rangeX = maxX - minX || 1;
        return width * 0.1 + (x - minX) / rangeX * width * 0.8;
    };
    const normalizeY = (y) => {
        const rangeY = maxY - minY || 1;
        return height * 0.85 - (y - minY) / rangeY * height * 0.7;
    };

    // 9. 准备绘图数据
    const dataset = features.map((sample, i) => ({
        x: normalizeX(sample[0]),
        y: normalizeY(sample[1]),
        cluster: labels[i]
    }));

    // 10. 生成颜色
    const colors = [];
    for (let i = 0; i < k; i++) {
        colors.push(`rgba(${Math.floor(Math.random()*255)}, ${Math.floor(Math.random()*255)}, ${Math.floor(Math.random()*255)}, 0.6)`);
    }

    // 11. 绘制聚类点
    dataset.forEach(point => {
        ctx.fillStyle = colors[point.cluster % colors.length];
        ctx.beginPath();
        ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#333';
        ctx.stroke();
    });

    // 12. 绘制聚类中心
    centroids.forEach((centroid, i) => {
        const x = normalizeX(centroid[0]);
        const y = normalizeY(centroid[1]);

        ctx.fillStyle = 'red';
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 2;
        ctx.stroke();
    });

    // 13. 绘制坐标轴
    ctx.beginPath();
    ctx.moveTo(width * 0.1, height * 0.85);
    ctx.lineTo(width * 0.9, height * 0.85);
    ctx.lineTo(width * 0.88, height * 0.83);
    ctx.moveTo(width * 0.9, height * 0.85);
    ctx.lineTo(width * 0.88, height * 0.87);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(width * 0.1, height * 0.85);
    ctx.lineTo(width * 0.1, height * 0.15);
    ctx.lineTo(width * 0.08, height * 0.17);
    ctx.moveTo(width * 0.1, height * 0.15);
    ctx.lineTo(width * 0.12, height * 0.17);
    ctx.stroke();

    // 14. 绘制标题和标签
    ctx.fillStyle = '#333';
    ctx.font = '16px Arial bold';
    ctx.textAlign = 'center';
    ctx.fillText('KMeans 聚类结果', width / 2, height * 0.1);

    ctx.font = '12px Arial';
    ctx.fillText('特征 0', width * 0.9, height * 0.88);
    ctx.save();
    ctx.translate(width * 0.07, height * 0.15);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('特征 1', 0, 0);
    ctx.restore();
}


// 绘制KNN
function drawKNN(ctx, width, height) {
    // 绘制数据点
    const points = [];
    const numPoints = 30;
    
    // 生成两类数据点
    for (let i = 0; i < numPoints; i++) {
        points.push({
            x: 150 + Math.random() * 150,
            y: 150 + Math.random() * 150,
            type: 0, // 类别0
            isNeighbor: false
        });
        
        points.push({
            x: 350 + Math.random() * 150,
            y: 300 + Math.random() * 150,
            type: 1, // 类别1
            isNeighbor: false
        });
    }
    
    // 新数据点
    const newPoint = {
        x: width / 2,
        y: height / 2,
        type: -1, // 未知类别
        isNeighbor: false
    };
    points.push(newPoint);
    
    // 计算与新点的距离并找出最近的5个点
    points.forEach(point => {
        if (point.type !== -1) {
            point.distance = Math.sqrt(
                Math.pow(point.x - newPoint.x, 2) + 
                Math.pow(point.y - newPoint.y, 2)
            );
        }
    });
    
    // 排序并标记最近的5个点
    const sortedPoints = points.filter(p => p.type !== -1).sort((a, b) => a.distance - b.distance);
    for (let i = 0; i < 5; i++) {
        sortedPoints[i].isNeighbor = true;
    }
    
    // 绘制所有点
    points.forEach(point => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, point.type === -1 ? 8 : 5, 0, Math.PI * 2);
        
        if (point.type === -1) {
            // 新点
            ctx.fillStyle = 'purple';
            ctx.fill();
            ctx.strokeStyle = 'black';
            ctx.lineWidth = 2;
            ctx.stroke();
        } else if (point.type === 0) {
            // 类别0
            ctx.fillStyle = 'blue';
            ctx.fill();
        } else {
            // 类别1
            ctx.fillStyle = 'red';
            ctx.fill();
        }
        
        // 标记最近邻点
        if (point.isNeighbor) {
            ctx.beginPath();
            ctx.arc(point.x, point.y, 10, 0, Math.PI * 2);
            ctx.strokeStyle = 'green';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    });
    
    // 绘制新点到最近邻点的连线
    sortedPoints.slice(0, 5).forEach(point => {
        ctx.beginPath();
        ctx.moveTo(newPoint.x, newPoint.y);
        ctx.lineTo(point.x, point.y);
        ctx.strokeStyle = 'green';
        ctx.lineWidth = 1;
        ctx.stroke();
    });
    
    // 绘制说明文本
    ctx.fillStyle = '#333';
    ctx.font = '14px Arial';
    ctx.textAlign = 'left';
    ctx.fillText('紫色: 待分类点', 50, 50);
    ctx.fillText('蓝色: 类别0', 50, 75);
    ctx.fillText('红色: 类别1', 50, 100);
    ctx.fillText('绿色标记: 最近的5个邻居', 50, 125);
}

// 绘制SVM
function drawSVM(ctx, width, height) {
    // 生成两类数据点
    const points = [];
    const numPoints = 20;
    
    // 类别A（上半部分）
    for (let i = 0; i < numPoints; i++) {
        points.push({
            x: 150 + Math.random() * 300,
            y: 300 + Math.random() * 100,
            type: 0
        });
    }
    
    // 类别B（下半部分）
    for (let i = 0; i < numPoints; i++) {
        points.push({
            x: 150 + Math.random() * 300,
            y: 100 + Math.random() * 100,
            type: 1
        });
    }
    
    // 绘制数据点
    points.forEach(point => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 6, 0, Math.PI * 2);
        ctx.fillStyle = point.type === 0 ? 'blue' : 'red';
        ctx.fill();
    });
    
    // 绘制最优超平面（一条直线）
    ctx.beginPath();
    ctx.moveTo(150, 200);
    ctx.lineTo(450, 200);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // 绘制间隔边界
    ctx.beginPath();
    ctx.moveTo(150, 150);
    ctx.lineTo(450, 150);
    ctx.strokeStyle = 'gray';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.moveTo(150, 250);
    ctx.lineTo(450, 250);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // 标记支持向量
    points.forEach(point => {
        if ((point.y > 140 && point.y < 160) || (point.y > 240 && point.y < 260)) {
            ctx.beginPath();
            ctx.arc(point.x, point.y, 10, 0, Math.PI * 2);
            ctx.strokeStyle = 'green';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    });
    
    // 绘制说明文本
    ctx.fillStyle = '#333';
    ctx.font = '14px Arial';
    ctx.textAlign = 'left';
    ctx.fillText('蓝色: 类别A', 50, 50);
    ctx.fillText('红色: 类别B', 50, 75);
    ctx.fillText('黑色实线: 最优超平面', 50, 100);
    ctx.fillText('灰色虚线: 间隔边界', 50, 125);
    ctx.fillText('绿色标记: 支持向量', 50, 150);
}

// 绘制随机森林
function drawRandomForest(ctx, width, height) {
    // 绘制多棵决策树
    const treePositions = [
        {x: width * 0.2, y: height * 0.3},
        {x: width * 0.5, y: height * 0.2},
        {x: width * 0.8, y: height * 0.35}
    ];
    
    treePositions.forEach(pos => {
        // 绘制简单的树结构
        drawNode(ctx, pos.x, pos.y, 60, 30, "特征");
        
        // 左分支
        drawLine(ctx, pos.x, pos.y + 15, pos.x - 30, pos.y + 50);
        drawNode(ctx, pos.x - 30, pos.y + 70, 50, 25, "是");
        
        // 右分支
        drawLine(ctx, pos.x, pos.y + 15, pos.x + 30, pos.y + 50);
        drawNode(ctx, pos.x + 30, pos.y + 70, 50, 25, "否");
    });
    
    // 绘制合并结果
    const mergeX = width / 2;
    const mergeY = height * 0.7;
    drawNode(ctx, mergeX, mergeY, 100, 40, "投票/平均");
    
    // 绘制最终结果
    drawLine(ctx, mergeX, mergeY + 20, mergeX, mergeY + 60);
    drawNode(ctx, mergeX, mergeY + 80, 80, 35, "最终结果");
    
    // 连接树到合并节点
    treePositions.forEach(pos => {
        drawLine(ctx, pos.x, pos.y + 85, mergeX, mergeY - 20);
    });
    
    // 绘制说明文本
    ctx.fillStyle = '#333';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('多棵决策树', width / 2, 30);
}

// 绘制线性回归
function drawLinearRegression(ctx, width, height) {
    // 生成随机数据点
    const points = [];
    const numPoints = 30;
    
    for (let i = 0; i < numPoints; i++) {
        const x = 100 + Math.random() * (width - 200);
        // 生成大致符合y = 0.5x + 50的点，加入随机噪声
        const y = height - (50 + 0.3 * (x - 100) + (Math.random() * 80 - 40));
        points.push({x, y});
    }
    
    // 绘制数据点
    points.forEach(point => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = 'blue';
        ctx.fill();
    });
    
    // 绘制回归线 (y = 0.5x + 50)
    const x1 = 100;
    const y1 = height - (50 + 0.3 * (x1 - 100));
    const x2 = width - 100;
    const y2 = height - (50 + 0.3 * (x2 - 100));
    
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // 绘制几条残差线
    const samplePoints = points.slice(0, 5);
    samplePoints.forEach(point => {
        // 计算点在回归线上的投影
        const regressionY = height - (50 + 0.3 * (point.x - 100));
        
        ctx.beginPath();
        ctx.moveTo(point.x, point.y);
        ctx.lineTo(point.x, regressionY);
        ctx.strokeStyle = 'green';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.stroke();
    });
    ctx.setLineDash([]);
    
    // 绘制公式
    ctx.fillStyle = '#333';
    ctx.font = '16px Arial';
    ctx.textAlign = 'left';
    ctx.fillText('y = wx + b', 50, 50);
    
    // 绘制说明文本
    ctx.font = '14px Arial';
    ctx.fillText('蓝色点: 数据点', 50, 80);
    ctx.fillText('红色线: 回归线', 50, 105);
    ctx.fillText('绿色虚线: 残差', 50, 130);
}

// 绘制逻辑回归
function drawLogisticRegression(ctx, width, height) {
    // 生成两类数据点
    const points = [];
    const numPoints = 20;
    
    // 类别0
    for (let i = 0; i < numPoints; i++) {
        points.push({
            x: 100 + Math.random() * 150,
            y: 100 + Math.random() * 200,
            type: 0
        });
    }
    
    // 类别1
    for (let i = 0; i < numPoints; i++) {
        points.push({
            x: 300 + Math.random() * 150,
            y: 100 + Math.random() * 200,
            type: 1
        });
    }
    
    // 绘制数据点
    points.forEach(point => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 6, 0, Math.PI * 2);
        ctx.fillStyle = point.type === 0 ? 'blue' : 'red';
        ctx.fill();
    });
    
    // 绘制决策边界
    ctx.beginPath();
    ctx.moveTo(250, 100);
    ctx.lineTo(250, 300);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // 绘制Sigmoid函数
    const sigmoidXStart = 100;
    const sigmoidXEnd = 400;
    const sigmoidYCenter = 350;
    const sigmoidHeight = 100;
    
    ctx.beginPath();
    for (let x = sigmoidXStart; x <= sigmoidXEnd; x++) {
        // Sigmoid函数: 1 / (1 + e^(-x))
        const t = (x - 250) / 50; // 缩放x值
        const sigmoid = 1 / (1 + Math.exp(-t));
        const y = sigmoidYCenter - sigmoid * sigmoidHeight;
        
        if (x === sigmoidXStart) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    ctx.strokeStyle = 'purple';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // 绘制坐标轴
    ctx.beginPath();
    ctx.moveTo(sigmoidXStart, sigmoidYCenter);
    ctx.lineTo(sigmoidXEnd, sigmoidYCenter);
    ctx.strokeStyle = 'gray';
    ctx.lineWidth = 1;
    ctx.stroke();
    
    ctx.beginPath();
    ctx.moveTo(250, sigmoidYCenter - sigmoidHeight);
    ctx.lineTo(250, sigmoidYCenter);
    ctx.stroke();
    
    // 标记0.5阈值
    ctx.fillStyle = '#333';
    ctx.font = '12px Arial';
    ctx.textAlign = 'right';
    ctx.fillText('0.5', 245, sigmoidYCenter - sigmoidHeight / 2 + 4);
    
    // 绘制公式
    ctx.font = '14px Arial';
    ctx.textAlign = 'left';
    ctx.fillText('σ(z) = 1 / (1 + e^(-z))', 50, 50);
    ctx.fillText('z = wx + b', 50, 75);
    
    // 绘制说明文本
    ctx.fillText('蓝色: 类别0', 50, 100);
    ctx.fillText('红色: 类别1', 50, 125);
    ctx.fillText('黑色线: 决策边界', 50, 150);
    ctx.fillText('紫色线: Sigmoid函数', 50, 175);
}

// 绘制AdaBoost
function drawAdaBoost(ctx, width, height) {
    // 绘制多个弱分类器
    const weakClassifiers = [
        {x: width * 0.2, y: height * 0.25},
        {x: width * 0.5, y: height * 0.25},
        {x: width * 0.8, y: height * 0.25}
    ];
    
    weakClassifiers.forEach((pos, index) => {
        drawNode(ctx, pos.x, pos.y, 80, 40, `弱分类器 ${index + 1}`);
        
        // 简单的决策边界示意
        ctx.beginPath();
        if (index === 0) {
            ctx.moveTo(pos.x - 30, pos.y + 60);
            ctx.lineTo(pos.x + 30, pos.y + 60);
        } else if (index === 1) {
            ctx.moveTo(pos.x - 30, pos.y + 40);
            ctx.lineTo(pos.x - 30, pos.y + 80);
        } else {
            ctx.moveTo(pos.x - 30, pos.y + 70);
            ctx.lineTo(pos.x + 30, pos.y + 50);
        }
        ctx.strokeStyle = 'gray';
        ctx.lineWidth = 1;
        ctx.stroke();
    });
    
    // 绘制加权组合
    const combineX = width / 2;
    const combineY = height * 0.55;
    drawNode(ctx, combineX, combineY, 120, 40, "加权组合");
    
    // 绘制强分类器
    const strongX = width / 2;
    const strongY = height * 0.75;
    drawNode(ctx, strongX, strongY, 100, 40, "强分类器");
    
    // 连接弱分类器到组合节点
    weakClassifiers.forEach((pos, index) => {
        drawLine(ctx, pos.x, pos.y + 40, combineX, combineY - 20);
        
        // 标记权重
        const weight = (0.3 + index * 0.1).toFixed(1);
        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`权重: ${weight}`, 
                   (pos.x + combineX) / 2, 
                   (pos.y + 40 + combineY - 20) / 2);
    });
    
    // 连接组合节点到强分类器
    drawLine(ctx, combineX, combineY + 20, strongX, strongY - 20);
    
    // 绘制说明文本
    ctx.fillStyle = '#333';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('多个弱分类器加权组合成强分类器', width / 2, 30);
}

// 绘制K均值聚类
function drawKMeans(ctx, width, height) {
    // 生成随机数据点
    const points = [];
    const numPoints = 50;
    const centers = [
        {x: 200, y: 200, color: 'red'},
        {x: 400, y: 200, color: 'blue'},
        {x: 300, y: 350, color: 'green'}
    ];
    const k = centers.length;
    
    // 围绕每个中心生成点
    centers.forEach(center => {
        for (let i = 0; i < numPoints / k; i++) {
            points.push({
                x: center.x + (Math.random() * 100 - 50),
                y: center.y + (Math.random() * 100 - 50),
                color: center.color
            });
        }
    });
    
    // 绘制数据点
    points.forEach(point => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = point.color;
        ctx.fill();
    });
    
    // 绘制聚类中心
    centers.forEach(center => {
        ctx.beginPath();
        ctx.arc(center.x, center.y, 10, 0, Math.PI * 2);
        ctx.fillStyle = center.color;
        ctx.fill();
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 2;
        ctx.stroke();
    });
    
    // 绘制聚类边界（简单示意）
    ctx.beginPath();
    ctx.ellipse(200, 200, 70, 70, 0, 0, Math.PI * 2);
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.ellipse(400, 200, 70, 70, 0, 0, Math.PI * 2);
    ctx.strokeStyle = 'blue';
    ctx.stroke();
    
    ctx.beginPath();
    ctx.ellipse(300, 350, 70, 70, 0, 0, Math.PI * 2);
    ctx.strokeStyle = 'green';
    ctx.stroke();
    ctx.setLineDash([]);
    
    // 绘制说明文本
    ctx.fillStyle = '#333';
    ctx.font = '14px Arial';
    ctx.textAlign = 'left';
    ctx.fillText('彩色点: 数据点', 50, 50);
    ctx.fillText('带黑边的点: 聚类中心', 50, 75);
    ctx.fillText('虚线: 聚类边界', 50, 100);
}

// 绘制EM算法
function drawEM(ctx, width, height) {
    // 绘制两个高斯分布的混合模型
    const center1 = {x: 200, y: 200};
    const center2 = {x: 400, y: 250};
    
    // 生成数据点
    const points = [];
    const numPoints = 60;
    
    // 来自第一个分布的点
    for (let i = 0; i < numPoints / 2; i++) {
        points.push({
            x: center1.x + (Math.random() * 120 - 60),
            y: center1.y + (Math.random() * 80 - 40),
            primary: 0
        });
    }
    
    // 来自第二个分布的点
    for (let i = 0; i < numPoints / 2; i++) {
        points.push({
            x: center2.x + (Math.random() * 100 - 50),
            y: center2.y + (Math.random() * 100 - 50),
            primary: 1
        });
    }
    
    // 绘制数据点（用渐变色表示属于两个分布的概率）
    points.forEach(point => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
        
        // 计算属于两个分布的概率（简单模拟）
        const d1 = Math.sqrt(Math.pow(point.x - center1.x, 2) + Math.pow(point.y - center1.y, 2));
        const d2 = Math.sqrt(Math.pow(point.x - center2.x, 2) + Math.pow(point.y - center2.y, 2));
        const p1 = 1 / (1 + d1 / d2); // 属于分布1的概率
        const p2 = 1 - p1; // 属于分布2的概率
        
        // 创建渐变色
        const gradient = ctx.createRadialGradient(
            point.x, point.y, 0,
            point.x, point.y, 5
        );
        gradient.addColorStop(0, `rgba(255, 0, 0, ${p1})`);
        gradient.addColorStop(1, `rgba(0, 0, 255, ${p2})`);
        
        ctx.fillStyle = gradient;
        ctx.fill();
    });
    
    // 绘制两个分布的轮廓
    ctx.beginPath();
    ctx.ellipse(center1.x, center1.y, 80, 50, 0, 0, Math.PI * 2);
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.ellipse(center2.x, center2.y, 70, 60, 0, 0, Math.PI * 2);
    ctx.strokeStyle = 'blue';
    ctx.stroke();
    ctx.setLineDash([]);
    
    // 绘制EM步骤
    const stepY = 380;
    drawNode(ctx, width * 0.3, stepY, 80, 40, "E步");
    drawNode(ctx, width * 0.7, stepY, 80, 40, "M步");
    drawLineWithArrow(ctx, width * 0.3 + 40, stepY, width * 0.7 - 40, stepY);
    drawLineWithArrow(ctx, width * 0.7, stepY + 20, width * 0.7, stepY + 60);
    drawLineWithArrow(ctx, width * 0.7, stepY + 80, width * 0.3, stepY + 80);
    drawLineWithArrow(ctx, width * 0.3, stepY + 80, width * 0.3, stepY + 20);
    
    // 步骤说明
    ctx.fillStyle = '#333';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('估计隐变量', width * 0.3, stepY + 60);
    ctx.fillText('更新模型参数', width * 0.7, stepY + 60);
    
    // 绘制说明文本
    ctx.font = '14px Arial';
    ctx.textAlign = 'left';
    ctx.fillText('红色: 分布1', 50, 50);
    ctx.fillText('蓝色: 分布2', 50, 75);
    ctx.fillText('渐变色点: 属于两个分布的概率', 50, 100);
}

// // 工具函数：绘制节点
// function drawNode(ctx, x, y, width, height, text) {
//     ctx.fillStyle = 'white';
//     ctx.fillRect(x - width / 2, y - height / 2, width, height);
    
//     ctx.strokeStyle = 'black';
//     ctx.lineWidth = 1;
//     ctx.strokeRect(x - width / 2, y - height / 2, width, height);
    
//     ctx.fillStyle = 'black';
//     ctx.font = '12px Arial';
//     ctx.textAlign = 'center';
//     ctx.textBaseline = 'middle';
//     ctx.fillText(text, x, y);
// }

// // 工具函数：绘制直线
// function drawLine(ctx, x1, y1, x2, y2) {
//     ctx.beginPath();
//     ctx.moveTo(x1, y1);
//     ctx.lineTo(x2, y2);
//     ctx.strokeStyle = 'black';
//     ctx.lineWidth = 1;
//     ctx.stroke();
// }

// 工具函数：绘制带箭头的直线
function drawLineWithArrow(ctx, x1, y1, x2, y2) {
    const headLength = 10;
    const angle = Math.atan2(y2 - y1, x2 - x1);
    
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 1;
    ctx.stroke();
    
    // 绘制箭头
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(
        x2 - headLength * Math.cos(angle - Math.PI / 6),
        y2 - headLength * Math.sin(angle - Math.PI / 6)
    );
    ctx.moveTo(x2, y2);
    ctx.lineTo(
        x2 - headLength * Math.cos(angle + Math.PI / 6),
        y2 - headLength * Math.sin(angle + Math.PI / 6)
    );
    ctx.stroke();
}

// 绘制数据集可视化

function drawIrisDataset(ctx, width, height, dataset) {
    const margin = 50;
    const plotWidth = width - 2 * margin;
    const plotHeight = height - 2 * margin;
    
    // 提取数据
    const samples = dataset.samples;  // 形状: [n, 4]
    const labels = dataset.labels;    // 形状: [n]
    
    // 提取花瓣长度(第2列)和花瓣宽度(第3列)
    const xData = samples.map(s => s[2]);
    const yData = samples.map(s => s[3]);
    
    // 计算最大最小值用于归一化
    const xMin = Math.min(...xData);
    const xMax = Math.max(...xData);
    const yMin = Math.min(...yData);
    const yMax = Math.max(...yData);
    
    // 绘制坐标轴
    ctx.beginPath();
    ctx.moveTo(margin, margin);
    ctx.lineTo(margin, height - margin);
    ctx.lineTo(width - margin, height - margin);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 1;
    ctx.stroke();
    
    // 绘制坐标轴标签
    ctx.fillStyle = '#333';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('花瓣长度 (cm)', width / 2, height - margin / 2);
    ctx.save();
    ctx.translate(margin / 2, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('花瓣宽度 (cm)', 0, 0);
    ctx.restore();
    
    // 绘制数据点
    const colors = ['red', 'green', 'blue'];
    samples.forEach((sample, i) => {
        const x = margin + (sample[2] - xMin) / (xMax - xMin) * plotWidth;
        const y = height - margin - (sample[3] - yMin) / (yMax - yMin) * plotHeight;
        
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = colors[labels[i]];
        ctx.fill();
    });
    
    // 绘制图例
    const classNames = ['山鸢尾', '变色鸢尾', '维吉尼亚鸢尾'];
    classNames.forEach((name, i) => {
        ctx.beginPath();
        ctx.arc(width - margin - 150, margin + 30 + i * 30, 5, 0, Math.PI * 2);
        ctx.fillStyle = colors[i];
        ctx.fill();
        
        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(name, width - margin - 140, margin + 34 + i * 30);
    });
}

function drawMNISTDataset(ctx, width, height, dataset) {
    const samples = dataset.samples;
    // const labels = dataset.labels;
    const labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    const digitSize = 50;
    const padding = 50;
    const cols = 5;
    const displayCount = Math.min(10, samples.length);
    
    for (let i = 0; i < displayCount; i++) {
        const col = i % cols;
        const row = Math.floor(i / cols);
        const x = 6.5 * padding + col * (digitSize + padding);
        const y = 4 * padding + row * (digitSize + padding);
        
        // 绘制框
        ctx.strokeStyle = '#ccc';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, digitSize, digitSize);
        
        // 绘制标签
        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(labels[i], x + digitSize / 2, y + digitSize + 20);
        
        // 绘制真实手写数字
        drawDigit(ctx, x+15, y+15, digitSize/2, labels[i]);
    }
    
    // 绘制标题
    ctx.fillStyle = '#333';
    ctx.font = '16px Arial bold';
    ctx.textAlign = 'center';
    ctx.fillText('MNIST手写数字样本', width / 2, padding);
}

// 绘制真实MNIST数字
function drawMNISTDigit(ctx, x, y, size, pixelData) {
    const canvas = ctx.canvas;
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = size;
    tempCanvas.height = size;
    const tempCtx = tempCanvas.getContext('2d');
    
    const imageData = tempCtx.createImageData(size, size);
    for (let i = 0; i < pixelData.length; i++) {
        const idx = i * 4;
        const value = 255 - pixelData[i]; // 反转颜色
        imageData.data[idx] = value;     // R
        imageData.data[idx + 1] = value; // G
        imageData.data[idx + 2] = value; // B
        imageData.data[idx + 3] = 255;   // A
    }
    
    tempCtx.putImageData(imageData, 0, 0);
    ctx.drawImage(tempCanvas, x, y);
}

function drawDigit(ctx, x, y, size, digit) {
    // 确保是整数
    const num = typeof digit === 'number' ? Math.floor(digit) : parseInt(digit, 10);

    ctx.fillStyle = 'black';
    ctx.strokeStyle = 'black';
    ctx.lineWidth = Math.max(2, size / 14);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    const unit = size / 14;

    ctx.beginPath();

    switch (num) {
        case 0:
            ctx.moveTo(x + unit * 2, y + unit);
            ctx.lineTo(x + size - unit * 2, y + unit);
            ctx.lineTo(x + size - unit * 2, y + size - unit);
            ctx.lineTo(x + unit * 2, y + size - unit);
            ctx.closePath();
            ctx.stroke();
            break;
        case 1:
            ctx.moveTo(x + size / 2, y + unit);
            ctx.lineTo(x + size / 2, y + size - unit);
            ctx.stroke();
            break;
        case 2:
            ctx.moveTo(x + unit * 2, y + unit * 2);
            ctx.lineTo(x + size - unit * 2, y + unit * 2);   // 上横
            ctx.lineTo(x + size - unit * 2, y + size / 2);  // 右竖
            ctx.lineTo(x + unit * 2, y + size / 2);         // 中横
            ctx.lineTo(x + unit * 2, y + size - unit * 2);  // 左竖
            ctx.lineTo(x + size - unit * 2, y + size - unit * 2); // 下横
            ctx.stroke();
            break;
        case 3:
            ctx.moveTo(x + unit * 2, y + unit * 2);
            ctx.lineTo(x + size - unit * 2, y + unit * 2);   // 上横
            ctx.lineTo(x + size - unit * 2, y + size / 2);  // 右上竖
            ctx.lineTo(x + unit * 2, y + size / 2);         // 中横
            ctx.lineTo(x +  size -unit * 2, y + size / 2);  // 左下竖
            ctx.lineTo(x + size - unit * 2, y + size -unit/ 2); // 下横
            ctx.lineTo(x - size/12 + unit * 2, y + size -unit/ 2); // 下横
            ctx.stroke();
            break;
        case 4:
            ctx.moveTo(x + unit * 2, y + unit * 2);
            ctx.lineTo(x + unit * 2, y + unit * 2+size/2);  // 左竖
            ctx.moveTo(x + unit * 2, y + size / 2);
            ctx.lineTo(x + size - unit * 2, y + size / 2);  // 中横
            ctx.moveTo(x + size - unit * 2, y + unit * 2);
            ctx.lineTo(x + size - unit * 2, y + size - unit * 2); // 右竖
            ctx.stroke();
            break;
        case 5:
            ctx.moveTo(x + size - unit * 2, y + unit * 2);
            ctx.lineTo(x + unit * 2, y + unit * 2);         // 上横
            ctx.lineTo(x + unit * 2, y + size / 2);         // 左竖
            ctx.lineTo(x + size - unit * 2, y + size / 2);  // 中横
            ctx.lineTo(x + size - unit * 2, y + size - unit * 2); // 右竖
            ctx.lineTo(x + unit * 2, y + size - unit * 2);  // 下横
            ctx.stroke();
            break;
        case 6:
            ctx.moveTo(x + size - unit * 2, y + unit * 2);
            ctx.lineTo(x + unit * 2, y + unit * 2);         // 上横
            ctx.lineTo(x + unit * 2, y + size - unit * 2);  // 左竖
            ctx.lineTo(x + size - unit * 2, y + size - unit * 2); // 下横
            ctx.lineTo(x + size - unit * 2, y + size / 2);  // 右竖
            ctx.lineTo(x + unit * 2, y + size / 2);         // 中横
            ctx.stroke();
            break;
        case 7:
            ctx.moveTo(x + unit * 2, y + unit * 2);
            ctx.lineTo(x + size - unit * 2, y + unit * 2);   // 上横
            ctx.lineTo(x + size - unit * 2, y + size - unit * 2); // 右竖
            ctx.stroke();
            break;
        case 8:
            ctx.moveTo(x + unit * 2, y + unit * 2);
            ctx.lineTo(x + size - unit * 2, y + unit * 2);   // 上横
            ctx.lineTo(x + size - unit * 2, y + size - unit * 2); // 右竖
            ctx.lineTo(x + unit * 2, y + size - unit * 2);  // 下横
            ctx.lineTo(x + unit * 2, y + unit * 2);         // 左竖
            ctx.moveTo(x + unit * 2, y + size / 2);
            ctx.lineTo(x + size - unit * 2, y + size / 2);  // 中横
            ctx.stroke();
            break;
        case 9:
            // 上半部分（近似椭圆用折线模拟）
            ctx.moveTo(x + unit * 4, y + unit * 2);
            ctx.lineTo(x + size - unit * 4, y + unit * 2);   // 上横
            ctx.lineTo(x + size - unit * 2, y + unit * 4);
            ctx.lineTo(x + size - unit * 2, y + size / 2 - unit); // 右上弧边
            ctx.lineTo(x + size - unit * 4, y + size / 2);
            ctx.lineTo(x + unit * 4, y + size / 2);             // 中横
            ctx.lineTo(x + unit * 2, y + size / 2 - unit);
            ctx.lineTo(x + unit * 2, y + unit * 4);
            ctx.closePath();
            
            // 右侧竖线
            ctx.moveTo(x + size - unit * 3, y + size / 2);
            ctx.lineTo(x + size - unit * 3, y + size - unit * 2);
            
            ctx.stroke();
            break;
    }
}

function drawRegressionDataset(ctx, width, height, dataset) {
    const margin = 50;
    const plotWidth = width - 2 * margin;
    const plotHeight = height - 2 * margin;
    
    const samples = dataset.samples;  // 形状: [n, 1]
    const labels = dataset.labels;    // 形状: [n]
    
    // 提取数据
    const xData = samples.map(s => s[0]);
    const yData = labels;
    
    // 计算最大最小值用于归一化
    const xMin = Math.min(...xData);
    const xMax = Math.max(...xData);
    const yMin = Math.min(...yData);
    const yMax = Math.max(...yData);
    
    // 绘制坐标轴
    ctx.beginPath();
    ctx.moveTo(margin, margin);
    ctx.lineTo(margin, height - margin);
    ctx.lineTo(width - margin, height - margin);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 1;
    ctx.stroke();
    
    // 绘制坐标轴标签
    ctx.fillStyle = '#333';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('特征X', width / 2, height - margin / 2);
    ctx.save();
    ctx.translate(margin / 2, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('目标值Y', 0, 0);
    ctx.restore();
    
    // 绘制数据点
    const points = [];
    samples.forEach((sample, i) => {
        const x = margin + (sample[0] - xMin) / (xMax - xMin) * plotWidth;
        const y = height - margin - (labels[i] - yMin) / (yMax - yMin) * plotHeight;
        
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = 'blue';
        ctx.fill();
        
        points.push({ x, y });
    });
    
    // 绘制说明
    ctx.fillStyle = '#333';
    ctx.font = '14px Arial';
    ctx.textAlign = 'left';
    ctx.fillText('蓝色点: 数据点', margin, margin - 10);
}

// 工具函数：绘制节点
function drawNode(ctx, x, y, width, height, text) {
    ctx.fillStyle = 'white';
    ctx.fillRect(x - width / 2, y - height / 2, width, height);
    
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 1;
    ctx.strokeRect(x - width / 2, y - height / 2, width, height);
    
    ctx.fillStyle = 'black';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, x, y);
}

// 工具函数：绘制直线
function drawLine(ctx, x1, y1, x2, y2) {
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 1;
    ctx.stroke();
}

// 工具函数：绘制带箭头的直线
function drawLineWithArrow(ctx, x1, y1, x2, y2) {
    const headLength = 10;
    const angle = Math.atan2(y2 - y1, x2 - x1);
    
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 1;
    ctx.stroke();
    
    // 绘制箭头
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(
        x2 - headLength * Math.cos(angle - Math.PI / 6),
        y2 - headLength * Math.sin(angle - Math.PI / 6)
    );
    ctx.moveTo(x2, y2);
    ctx.lineTo(
        x2 - headLength * Math.cos(angle + Math.PI / 6),
        y2 - headLength * Math.sin(angle + Math.PI / 6)
    );
    ctx.stroke();
}
