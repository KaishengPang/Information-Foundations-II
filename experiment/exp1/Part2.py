import numpy as np
import matplotlib.pyplot as plt

# 修改为tanh激活函数及其导数，这里实际使用了tanh激活函数
def tanh(x):
    return np.tanh(x)  # tanh函数：输出范围在-1到1之间

# tanh激活函数的导数
def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2  # tanh导数：用于反向传播时调整权重

# 初始化神经网络的权重和偏置
def initialize_network(input_size, hidden_size, output_size):
    np.random.seed(42)  # 固定随机种子，确保每次运行的初始权重相同，便于调试和比较结果

    # 初始化输入层到隐藏层的权重矩阵，权重值在-1到1之间，维度为(input_size, hidden_size)
    W_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))

    # 初始化隐藏层到输出层的权重矩阵，维度为(hidden_size, output_size)
    W_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))

    # 初始化隐藏层的偏置，维度为(1, hidden_size)
    b_hidden = np.random.uniform(-1, 1, (1, hidden_size))

    # 初始化输出层的偏置，维度为(1, output_size)
    b_output = np.random.uniform(-1, 1, (1, output_size))

    return W_input_hidden, W_hidden_output, b_hidden, b_output  # 返回初始化的权重和偏置

# 前向传播计算过程
def forward_propagation(X, W_input_hidden, W_hidden_output, b_hidden, b_output):
    # 计算隐藏层的输入：X * W_input_hidden + b_hidden
    hidden_input = np.dot(X, W_input_hidden) + b_hidden

    # 使用tanh激活函数计算隐藏层的输出
    hidden_output = tanh(hidden_input)  # 隐藏层激活函数为tanh

    # 计算输出层的输入：隐藏层输出 * W_hidden_output + b_output
    final_input = np.dot(hidden_output, W_hidden_output) + b_output

    # 输出层使用线性激活函数，即输出层没有非线性转换，直接输出
    final_output = final_input

    return hidden_output, final_output  # 返回隐藏层的输出和最终的输出

# 损失函数：均方误差
def compute_loss(y_true, y_pred):
    return 0.5 * np.mean((y_true - y_pred) ** 2)  # 计算预测值与真实值的均方误差

# 反向传播并使用Adam优化器更新权重
def backward_propagation_adam(X, y, hidden_output, final_output, W_hidden_output, W_input_hidden, b_output, b_hidden,
                              m_W_hidden_output, v_W_hidden_output, m_W_input_hidden, v_W_input_hidden,
                              m_b_output, v_b_output, m_b_hidden, v_b_hidden, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    # 计算输出层的误差：真实值与预测值的差异
    output_error = y - final_output
    # 输出层没有激活函数，误差直接作为delta
    output_delta = output_error

    # 计算隐藏层的误差：输出层的delta乘以隐藏层到输出层的权重矩阵的转置
    hidden_error = output_delta.dot(W_hidden_output.T)
    # 计算隐藏层的delta：隐藏层误差乘以tanh激活函数的导数
    hidden_delta = hidden_error * tanh_derivative(hidden_output)

    # 使用Adam优化器更新权重和偏置，按Adam的公式进行动量更新和二阶矩更新
    for param, grad, m, v in zip([W_input_hidden, W_hidden_output, b_hidden, b_output],
                                 [X.T.dot(hidden_delta), hidden_output.T.dot(output_delta), np.sum(hidden_delta, axis=0, keepdims=True),
                                  np.sum(output_delta, axis=0, keepdims=True)],
                                 [m_W_input_hidden, m_W_hidden_output, m_b_hidden, m_b_output],
                                 [v_W_input_hidden, v_W_hidden_output, v_b_hidden, v_b_output]):

        # 一阶矩m的更新
        m[:] = beta1 * m + (1 - beta1) * grad
        # 二阶矩v的更新
        v[:] = beta2 * v + (1 - beta2) * grad**2

        # 偏差修正后的m和v
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        # 使用Adam更新参数（权重或偏置）
        param += learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    return W_input_hidden, W_hidden_output, b_hidden, b_output, m_W_hidden_output, v_W_hidden_output, m_W_input_hidden, v_W_input_hidden, m_b_output, v_b_output, m_b_hidden, v_b_hidden

# 训练神经网络（使用Adam优化器）
def train(X, y, input_size, hidden_size, output_size, epochs=10000, learning_rate=0.01):
    # 初始化网络的权重和偏置
    W_input_hidden, W_hidden_output, b_hidden, b_output = initialize_network(input_size, hidden_size, output_size)

    # 初始化Adam优化器的动量和二阶矩
    m_W_input_hidden, v_W_input_hidden = np.zeros_like(W_input_hidden), np.zeros_like(W_input_hidden)
    m_W_hidden_output, v_W_hidden_output = np.zeros_like(W_hidden_output), np.zeros_like(W_hidden_output)
    m_b_hidden, v_b_hidden = np.zeros_like(b_hidden), np.zeros_like(b_hidden)
    m_b_output, v_b_output = np.zeros_like(b_output), np.zeros_like(b_output)

    # 开始迭代训练，进行多次前向和反向传播
    for epoch in range(epochs):
        # 前向传播，计算隐藏层和输出层的输出
        hidden_output, final_output = forward_propagation(X, W_input_hidden, W_hidden_output, b_hidden, b_output)
        # 计算当前的损失值
        loss = compute_loss(y, final_output)

        # 反向传播，并使用Adam优化器更新权重和偏置
        W_input_hidden, W_hidden_output, b_hidden, b_output, m_W_hidden_output, v_W_hidden_output, m_W_input_hidden, v_W_input_hidden, \
        m_b_output, v_b_output, m_b_hidden, v_b_hidden = backward_propagation_adam(X, y, hidden_output, final_output, W_hidden_output,
                                                                                  W_input_hidden, b_output, b_hidden,
                                                                                  m_W_hidden_output, v_W_hidden_output,
                                                                                  m_W_input_hidden, v_W_input_hidden,
                                                                                  m_b_output, v_b_output,
                                                                                  m_b_hidden, v_b_hidden, epoch + 1,
                                                                                  learning_rate)

        # 每1000次迭代打印一次损失值
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

        # 如果损失小于1e-5，提前终止训练
        if loss < 1e-5:
            print(f"Training stopped at epoch {epoch}")
            break

    return W_input_hidden, W_hidden_output, b_hidden, b_output  # 返回训练后的网络参数

# 定义输入数据范围
X = np.linspace(-np.pi/2+0.05, -0.05, 200).reshape(-1, 1)  # x从-pi/2到0的部分
X = np.concatenate([X, np.linspace(0.05, np.pi/2-0.05, 200).reshape(-1, 1)])  # x从0到pi/2的部分

# 定义目标输出：函数y = 1 / sin(x) + 1 / cos(x)
y = 1 / np.sin(X) + 1 / np.cos(X)

# 设置神经网络的参数
input_size = 1  # 输入维度为1
hidden_size = 120  # 隐藏层包含120个神经元
output_size = 1  # 输出维度为1
learning_rate = 0.001  # 学习率
epochs = 100000  # 最大训练迭代次数

# 训练神经网络
W_input_hidden, W_hidden_output, b_hidden, b_output = train(X, y, input_size, hidden_size, output_size, epochs, learning_rate)

# 使用训练好的模型进行预测
hidden_output, final_output = forward_propagation(X, W_input_hidden, W_hidden_output, b_hidden, b_output)
print("Final output after training:")
print(final_output)

# 绘制真实值与预测值的对比图
plt.figure(figsize=(10, 6))

# 绘制真实函数曲线
plt.scatter(X, y, label="True Function", color="blue")

# 绘制神经网络的预测输出
plt.scatter(X, final_output, label="Neural Network Prediction", color="red")

# 限制y轴的范围
plt.ylim(-20,20)

# 设置图表标题和轴标签
plt.title("True Function vs Neural Network Prediction (Scatter Plot)")
plt.xlabel("X")
plt.ylabel("Y")

# 显示图例
plt.legend()

# 显示网格
plt.grid(True)

# 显示图像
plt.show()
#-----------------------------------------探究不同学习率的影响--------------------------------------------------------------
# # 定义不同学习率的列表
# learning_rates = [0.001, 0.005, 0.01, 0.1]
# epochs = 10000  # 固定迭代次数
#
# # 存储不同学习率下的损失变化
# loss_histories = {}
#
# # 针对不同学习率进行实验
# for lr in learning_rates:
#     print(f"Training with learning rate: {lr}")
#     # 训练神经网络
#     W_input_hidden, W_hidden_output, b_hidden, b_output = train(X, y, input_size, hidden_size, output_size, epochs,
#                                                                 learning_rate=lr)
#
#     # 使用训练好的模型进行预测
#     hidden_output, final_output = forward_propagation(X, W_input_hidden, W_hidden_output, b_hidden, b_output)
#
#     # 计算训练损失
#     loss_histories[lr] = []
#     for epoch in range(epochs):
#         hidden_output, final_output = forward_propagation(X, W_input_hidden, W_hidden_output, b_hidden, b_output)
#         loss = compute_loss(y, final_output)
#         loss_histories[lr].append(loss)

#-----------------------------------------探究不同权值的影响---------------------------------------------------------------
# 人为修改部分连接权重
# def modify_weights(W_input_hidden, W_hidden_output, modification_rate=0.1):
#     W_input_hidden_modified = W_input_hidden + np.random.uniform(-modification_rate, modification_rate, W_input_hidden.shape)
#     W_hidden_output_modified = W_hidden_output + np.random.uniform(-modification_rate, modification_rate, W_hidden_output.shape)
#     return W_input_hidden_modified, W_hidden_output_modified
#
#
# # 训练后的预测结果
# _, original_output = forward_propagation(X, W_input_hidden, W_hidden_output, b_hidden, b_output)
#
# # 修改权重
# W_input_hidden_modified, W_hidden_output_modified = modify_weights(W_input_hidden, W_hidden_output)
#
# # 修改权重后的预测结果
# _, modified_output = forward_propagation(X, W_input_hidden_modified, W_hidden_output_modified, b_hidden, b_output)
#
# # 绘制修改前后的预测对比图
# plt.figure(figsize=(10, 6))
#
# # 绘制真实函数曲线
# plt.scatter(X, y, label="True Function", color="blue")
#
# # 绘制修改前的预测结果
# plt.scatter(X, original_output, label="Prediction Before Weight Modification", color="green")
#
# # 绘制修改后的预测结果
# plt.scatter(X, modified_output, label="Prediction After Weight Modification", color="purple")
#
# # 限制y轴的范围
# plt.ylim(-20, 20)
#
# # 设置图表标题和轴标签
# plt.title("Comparison of Predictions Before and After Weight Modification")
# plt.xlabel("X")
# plt.ylabel("Y")
#
# # 显示图例
# plt.legend()
#
# # 显示网格
# plt.grid(True)
#
# # 显示图像
# plt.show()