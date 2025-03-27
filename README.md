# 逻辑回归实现

这个项目使用NumPy从零开始构建了一个完整的二分类逻辑回归模型。通过这个实现，可以深入理解逻辑回归的原理和实现细节，包括梯度下降优化、正则化和模型评估等关键概念。

## 功能特点

- 从零实现二分类逻辑回归算法
- 支持L1和L2正则化（Lasso和Ridge）
- 完整的梯度下降优化过程
- 提供数据可视化工具：决策边界、概率分布和损失函数变化
- 模型评估：准确率和混淆矩阵
- 参数归一化和还原功能

## 算法原理

### 逻辑回归模型
逻辑回归使用Sigmoid函数将线性模型的输出映射到[0,1]区间，表示样本属于正类的概率:

```
P(y=1|x) = sigmoid(wx + b) = 1 / (1 + e^-(wx + b))
```

### 损失函数
使用交叉熵损失函数（对数似然损失）:

```
L = -1/m * ∑[y*log(p) + (1-y)*log(1-p)]
```

### 正则化
为防止过拟合，支持两种正则化方式:

- **L1正则化(Lasso)**: 添加 `λ * ∑|w_i|` 到损失函数，倾向于产生稀疏权重
- **L2正则化(Ridge)**: 添加 `λ * ∑w_i²` 到损失函数，倾向于减小所有权重的幅度

## 使用方法

### 基本用法

```python
python logistic_regression.py
```

### 参数调整
可在main函数中调整以下参数:

- `learning_rate`: 学习率，控制每次参数更新的步长（默认0.1）
- `num_iterations`: 训练迭代次数（默认5000）
- `lambda_`: 正则化强度，较大的值增强正则化效果（默认0.01）
- `penalty`: 正则化类型，可选 'l1' 或 'l2'（默认'l2'）

### 代码示例

```python
# 初始化参数
W = np.zeros(2)
b = 0
learning_rate = 0.1
num_iterations = 5000
lambda_ = 0.01  # 正则化参数
penalty = 'l2'  # 'l1'或'l2'

# 训练模型
W, b, cost_history, W_history, b_history = train(
    X_norm, Y, W, b, learning_rate, num_iterations, lambda_, penalty
)
```

## 输出解释

### 训练过程
训练过程中会每1000次迭代输出一次损失值和准确率:

```
迭代 1/5000, 损失: 0.6932, 准确率: 0.5020
迭代 1000/5000, 损失: 0.3256, 准确率: 0.8540
...
```

### 训练结果
训练完成后会输出:

1. 原始尺度的权重和偏置
2. 正则化类型和参数
3. 模型准确率
4. 混淆矩阵

### 可视化输出

程序会生成三个可视化图表:

1. **损失函数历史**: 展示训练过程中损失函数的变化
2. **决策边界**: 在特征空间中显示模型的决策边界和训练样本
3. **概率分布**: 3D图展示模型对特征空间中每个点的概率预测

## 依赖库

- NumPy: 数值计算
- Matplotlib: 数据可视化

## 实现细节

- Sigmoid函数用于将线性输出转换为概率
- 使用交叉熵损失函数评估模型性能
- 基于梯度下降优化参数
- 特征归一化提高训练效率和稳定性
- 正则化技术防止过拟合
- 混淆矩阵用于全面评估分类性能

## 函数说明

- `sigmoid(z)`: 计算Sigmoid函数
- `generate_samples(num_samples)`: 生成二分类样本数据
- `predict(X, W, b)`: 计算每个样本的预测概率
- `predict_class(X, W, b, threshold=0.5)`: 将概率转换为类别预测
- `loss(X, Y, W, b, lambda_=0, penalty='l2')`: 计算带正则化的交叉熵损失
- `gradient(X, Y, W, b, lambda_=0, penalty='l2')`: 计算损失函数的梯度
- `train(X, Y, W, b, learning_rate, num_iterations, lambda_=0, penalty='l2')`: 训练模型
- `normalize(X)`: 特征归一化
- `denormalize_parameters(W, b, mean_X, std_X)`: 将参数还原到原始尺度
- `confusion_matrix(y_true, y_pred)`: 计算混淆矩阵
