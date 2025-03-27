import numpy as np
import matplotlib.pyplot as plt

from matplotlib import font_manager
# macOS系统中文字体设置
plt.rcParams['font.family'] = ['Arial Unicode MS']  # macOS自带Unicode字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号



#生成样本,f_x=5x1+7x2+9
def generate_samples(num_samples):
    x1 = 3 * np.random.randn(num_samples) + 10
    x2 = 6 *np.random.randn(num_samples) + 2
    noise = np.random.randn(num_samples) 
    x = np.column_stack((x1, x2))
    y = 5 * x1 + 7 * x2 + 9 + noise
    return x,y
#预测
def predict(X,W,b):
    return np.matmul(X, W) + b
#误差
def error(X,Y,W,b):
    return Y - predict(X,W,b)
#损失函数
def loss(X,Y,W,b):
    return np.mean(error(X,Y,W,b)**2)
#成本函数
def cost_function(X,Y,W,b):
    return  1/2 * loss(X,Y,W,b)
#梯度
def gradient(X,Y,W,b):
    n = len(Y)
    err = error(X,Y,W,b)
    dW = -2/n * np.matmul(X.T, err)
    db = -2/n * np.sum(err)
    return dW, db
#更新参数
def update_parameters(X,Y,W,b,learning_rate):
    dW,db = gradient(X,Y,W,b)
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W,b
#训练
def train(X,Y,W,b,learning_rate,num_iterations):
    W_history = []
    b_history = []
    cost_history = []
    for i in range(num_iterations):
        W,b = update_parameters(X,Y,W,b,learning_rate)
        W_history.append(W)
        b_history.append(b)
        cost_history.append(cost_function(X,Y,W,b))
        print(f"Iteration {i+1}/{num_iterations}, Cost: {cost_history[-1]},W:{W},b:{b}")
    return W,b,cost_history,W_history,b_history

def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std

def denormalize_parameters(W, b, mean_X, std_X, mean_Y, std_Y):
    """
    将归一化后训练得到的参数还原回原始尺度
    
    参数:
    W - 归一化特征训练的权重
    b - 归一化特征训练的偏置
    mean_X - 特征均值
    std_X - 特征标准差
    mean_Y - 标签均值
    std_Y - 标签标准差
    
    返回:
    W_original - 原始尺度的权重
    b_original - 原始尺度的偏置
    """
    # 还原权重
    W_original = W * std_Y / std_X
    
    # 还原偏置
    b_original = b * std_Y + mean_Y - np.sum(W_original * mean_X)
    
    return W_original, b_original

def main():
    X,Y = generate_samples(100)
    X_norm, mean_X, std_X = normalize(X)
    Y_norm, mean_Y, std_Y = normalize(Y)
    W = np.zeros(2)
    b = 0
    learning_rate = 0.01 #学习率
    num_iterations = 500 #迭代次数
    W,b,cost_history,W_history,b_history = train(X_norm,Y_norm,W,b,learning_rate,num_iterations)
    
    # 还原参数到原始尺度
    W_original, b_original = denormalize_parameters(W, b, mean_X, std_X, mean_Y, std_Y)
    print(f"原始尺度参数 - W: {W_original}, b: {b_original}")
    
    # 验证还原后的参数
    y_pred_norm = predict(X_norm, W, b)
    y_pred_original = predict(X, W_original, b_original)
    mse_norm = np.mean((Y_norm - y_pred_norm)**2)
    mse_original = np.mean((Y - y_pred_original)**2)
    print(f"归一化数据MSE: {mse_norm}")
    print(f"原始数据MSE: {mse_original}")

    # 可视化
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(num_iterations), cost_history)
    plt.title('损失函数历史')
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X[:,0], Y, alpha=0.5, label='实际值')
    plt.scatter(X[:,0], predict(X, W_original, b_original), alpha=0.5, label='预测值')
    plt.legend()
    plt.title('预测结果')
    plt.show()

if __name__ == "__main__":
    main()

