import numpy as np
import matplotlib.pyplot as plt

from matplotlib import font_manager
# macOS系统中文字体设置
plt.rcParams['font.family'] = ['Arial Unicode MS']  # macOS自带Unicode字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# Sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 生成二分类样本
def generate_samples(num_samples):
    x1 = 3 * np.random.randn(num_samples) + 10
    x2 = 6 * np.random.randn(num_samples) + 2
    x = np.column_stack((x1, x2))
    
    # 生成二分类标签
    z = 5 * x1 + 7 * x2 + 9 + np.random.randn(num_samples) * 5
    y = (z > np.median(z)).astype(int)  # 大于中位数为1，否则为0
    
    return x, y

# 预测概率
def predict(X, W, b):
    z = np.matmul(X, W) + b
    return sigmoid(z)

# 二分类预测
def predict_class(X, W, b, threshold=0.5):
    return (predict(X, W, b) >= threshold).astype(int)

# 交叉熵损失函数（带正则化）
def loss(X, Y, W, b, lambda_=0, penalty='l2'):
    m = len(Y)
    y_pred = predict(X, W, b)
    # 避免log(0)导致的数值问题
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # 交叉熵损失
    cross_entropy = -1/m * np.sum(Y * np.log(y_pred) + (1 - Y) * np.log(1 - y_pred))
    
    # 添加正则化项
    if lambda_ > 0:
        if penalty == 'l2':  # L2正则化 (Ridge)
            reg_term = lambda_ * np.sum(W**2) / (2*m)
        elif penalty == 'l1':  # L1正则化 (Lasso)
            reg_term = lambda_ * np.sum(np.abs(W)) / m
        else:
            raise ValueError("正则化类型必须是'l1'或'l2'")
        
        return cross_entropy + reg_term
    return cross_entropy

# 成本函数（与损失函数相同）
def cost_function(X, Y, W, b, lambda_=0, penalty='l2'):
    return loss(X, Y, W, b, lambda_, penalty)

# 梯度（带正则化）
def gradient(X, Y, W, b, lambda_=0, penalty='l2'):
    m = len(Y)
    y_pred = predict(X, W, b)
    error = y_pred - Y
    
    # 基本梯度
    dW = 1/m * np.matmul(X.T, error)
    db = 1/m * np.sum(error)
    
    # 添加正则化项的梯度
    if lambda_ > 0:
        if penalty == 'l2':  # L2正则化 (Ridge)
            dW += (lambda_ / m) * W
        elif penalty == 'l1':  # L1正则化 (Lasso)
            dW += (lambda_ / m) * np.sign(W)
    
    return dW, db

# 更新参数
def update_parameters(X, Y, W, b, learning_rate, lambda_=0, penalty='l2'):
    dW, db = gradient(X, Y, W, b, lambda_, penalty)
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b

# 训练
def train(X, Y, W, b, learning_rate, num_iterations, lambda_=0, penalty='l2'):
    W_history = []
    b_history = []
    cost_history = []
    for i in range(num_iterations):
        W, b = update_parameters(X, Y, W, b, learning_rate, lambda_, penalty)
        W_history.append(W)
        b_history.append(b)
        cost_history.append(cost_function(X, Y, W, b, lambda_, penalty))
        
        if (i+1) % 1000 == 0 or i == 0:
            y_pred_class = predict_class(X, W, b)
            accuracy = np.mean(y_pred_class == Y)
            print(f"迭代 {i+1}/{num_iterations}, 损失: {cost_history[-1]:.4f}, 准确率: {accuracy:.4f}")
            
    return W, b, cost_history, W_history, b_history

def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std

def denormalize_parameters(W, b, mean_X, std_X):
    """
    将归一化后训练得到的参数还原回原始尺度
    
    参数:
    W - 归一化特征训练的权重
    b - 归一化特征训练的偏置
    mean_X - 特征均值
    std_X - 特征标准差
    
    返回:
    W_original - 原始尺度的权重
    b_original - 原始尺度的偏置
    """
    # 还原权重和偏置
    W_original = W / std_X
    b_original = b - np.sum(W_original * mean_X)
    
    return W_original, b_original

# 计算混淆矩阵
def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    return np.array([[TN, FP], [FN, TP]])

def main():
    # 生成数据
    X, Y = generate_samples(1000)
    
    # 归一化特征
    X_norm, mean_X, std_X = normalize(X)
    
    # 初始化参数
    W = np.zeros(2)
    b = 0
    learning_rate = 0.1  # 学习率
    num_iterations = 5000  # 迭代次数
    lambda_ = 0.01  # 正则化参数
    penalty = 'l2'  # 'l1'或'l2'
    
    # 训练模型
    W, b, cost_history, W_history, b_history = train(X_norm, Y, W, b, learning_rate, num_iterations, lambda_, penalty)
    
    # 还原参数到原始尺度
    W_original, b_original = denormalize_parameters(W, b, mean_X, std_X)
    print(f"原始尺度参数 - W: {W_original}, b: {b_original}")
    print(f"正则化类型: {penalty}, 正则化参数: {lambda_}")
    
    # 评估模型
    y_pred = predict_class(X_norm, W, b)
    accuracy = np.mean(y_pred == Y)
    cm = confusion_matrix(Y, y_pred)
    
    print(f"准确率: {accuracy:.4f}")
    print(f"混淆矩阵: \n{cm}")
    
    # 可视化
    plt.figure(figsize=(15, 5))
    
    # 损失历史
    plt.subplot(1, 3, 1)
    plt.plot(range(num_iterations), cost_history)
    plt.title('损失函数历史')
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    
    # 决策边界
    plt.subplot(1, 3, 2)
    
    # 创建网格
    h = 0.1
    x_min, x_max = X_norm[:, 0].min() - 1, X_norm[:, 0].max() + 1
    y_min, y_max = X_norm[:, 1].min() - 1, X_norm[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # 预测网格点的类别
    Z = predict_class(np.c_[xx.ravel(), yy.ravel()], W, b)
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    
    # 绘制训练样本
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors='k')
    plt.title('决策边界')
    plt.xlabel('特征1 (归一化)')
    plt.ylabel('特征2 (归一化)')
    
    # 概率分布
    plt.subplot(1, 3, 3, projection='3d')
    ax = plt.gca()
    
    # 创建更小的网格以提高性能
    h = 0.2
    xx_small, yy_small = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # 预测概率
    Z_prob = predict(np.c_[xx_small.ravel(), yy_small.ravel()], W, b)
    Z_prob = Z_prob.reshape(xx_small.shape)
    
    # 绘制概率曲面
    surf = ax.plot_surface(xx_small, yy_small, Z_prob, cmap=plt.cm.coolwarm, alpha=0.6)
    
    # 绘制决策边界（概率=0.5）
    ax.contour(xx_small, yy_small, Z_prob, [0.5], colors='k', linestyles='solid')
    
    # 添加样本点
    ax.scatter(X_norm[Y==0, 0], X_norm[Y==0, 1], np.zeros(np.sum(Y==0)), c='b', marker='o', label='类别0')
    ax.scatter(X_norm[Y==1, 0], X_norm[Y==1, 1], np.ones(np.sum(Y==1)), c='r', marker='^', label='类别1')
    
    ax.set_xlabel('特征1 (归一化)')
    ax.set_ylabel('特征2 (归一化)')
    ax.set_zlabel('预测概率')
    ax.set_title('概率分布与决策边界')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

