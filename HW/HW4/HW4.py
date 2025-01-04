import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成数据
np.random.seed(42)  # 设置随机种子以确保结果可重复
mean_A = [0, 0]  # A 类的均值
cov_A = [[1, 0], [0, 1]]  # A 类的协方差矩阵
mean_B = [2, 3]  # B 类的均值
cov_B = [[3, 0], [0, 4]]  # B 类的协方差矩阵

# 随机生成 30 个 A 类点和 30 个 B 类点
A_points = np.random.multivariate_normal(mean_A, cov_A, 30)
B_points = np.random.multivariate_normal(mean_B, cov_B, 30)

# 创建标签
A_labels = np.ones(30) * -1  # A 类标签为 -1
B_labels = np.ones(30)  # B 类标签为 1

# 合并数据和标签
X = np.vstack((A_points, B_points))  # 将 A 类和 B 类的数据垂直堆叠在一起
y = np.hstack((A_labels, B_labels))  # 将 A 类和 B 类的标签水平堆叠在一起

# 转换为 torch 张量
X = torch.tensor(X, dtype=torch.float32)  # 转换数据为 torch 的浮点张量
y = torch.tensor(y, dtype=torch.float32)  # 转换标签为 torch 的浮点张量

# 2. 计算 RBF 核矩阵
def rbf_kernel(X1, X2, gamma):
    sq_dist = torch.cdist(X1, X2, p=2) ** 2  # 计算欧式距离的平方
    return torch.exp(-gamma * sq_dist)  # 计算 RBF 核矩阵

gamma = 0.1  # RBF 核的参数
K = rbf_kernel(X, X, gamma=gamma)  # 计算训练数据的核矩阵

# 3. 定义 SVM 模型
class SVM(torch.nn.Module):
    def __init__(self):
        super(SVM, self).__init__()
        self.alpha = torch.nn.Parameter(torch.randn(K.shape[0], requires_grad=True))  # 初始化拉格朗日乘子 alpha
        self.b = torch.nn.Parameter(torch.zeros(1, requires_grad=True))  # 初始化偏置项 b

    def forward(self, kernel_matrix):
        return torch.matmul(kernel_matrix, self.alpha) + self.b  # SVM 的输出，线性组合核函数和偏置项

# 4. 定义损失函数（Hinge Loss）
def hinge_loss(output, target):
    return torch.mean(torch.clamp(1 - output * target, min=0))  # Hinge Loss，用于最大化分类间隔

# 5. 训练 SVM
model = SVM()  # 实例化 SVM 模型
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用 Adam 优化器，学习率为 0.01

# 训练循环
epochs = 3000  # 训练次数
losses = []  # 存储每个 epoch 的损失值
for epoch in range(epochs):
    # 前向传播
    output = model(K)  # 计算当前模型对所有数据点的输出
    
    # 计算损失
    loss = hinge_loss(output, y)  # 使用 Hinge Loss 计算损失
    losses.append(loss.item())  # 记录当前的损失值
    
    # 反向传播和优化
    optimizer.zero_grad()  # 清除上一步的梯度
    loss.backward()  # 计算当前的梯度
    optimizer.step()  # 使用优化器更新模型参数
    
    # 打印损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')  # 每 100 次迭代打印一次损失值

# 6. 绘制损失曲线
plt.plot(range(epochs), losses)  # 绘制损失值随 epoch 变化的曲线
plt.xlabel('Epochs')  # 横坐标为训练次数
plt.ylabel('Loss')  # 纵坐标为损失值
plt.title('Loss Curve for SVM Training')  # 图的标题
plt.show()  # 显示损失曲线

# 7. 计算准确率函数
def calculate_accuracy(model, K, y):
    # 前向传播，获取模型的预测输出
    output = model(K).detach().numpy()  # 计算输出并转换为 numpy 数组
    
    # 将输出结果转换为预测类别，>= 0 的为 1 类，< 0 的为 -1 类
    predictions = np.where(output >= 0, 1, -1)
    
    # 计算预测正确的数量
    correct_predictions = np.sum(predictions == y.numpy())
    
    # 计算准确率
    accuracy = correct_predictions / len(y)
    return accuracy

# 在训练完成后，计算训练集上的准确率
accuracy = calculate_accuracy(model, K, y)
print(f'Training Accuracy: {accuracy * 100:.2f}%')

# 8. 绘制分类结果
alpha = model.alpha.detach().numpy()  # 获取训练好的 alpha 参数
b = model.b.item()  # 获取训练好的偏置项 b

# 创建网格以绘制决策边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # 网格的 X 轴范围
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # 网格的 Y 轴范围
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))  # 创建网格

grid = np.c_[xx.ravel(), yy.ravel()]  # 将网格点展开为二维数组
grid_tensor = torch.tensor(grid, dtype=torch.float32)  # 转换为 torch 张量

# 计算网格点的核矩阵并预测类别
K_grid = rbf_kernel(grid_tensor, X, gamma=gamma)  # 计算网格点和训练数据之间的核矩阵
Z = model.forward(K_grid).detach().numpy()  # 计算每个网格点的输出值，并转换为 numpy 数组
Z = Z.reshape(xx.shape)  # 重塑为网格的形状

# 绘制原始数据点
plt.scatter(A_points[:, 0], A_points[:, 1], color='red', label='Class A')  # 绘制 A 类点，红色
plt.scatter(B_points[:, 0], B_points[:, 1], color='blue', label='Class B')  # 绘制 B 类点，蓝色

# 绘制决策边界
plt.contourf(xx, yy, Z, levels=[-float('inf'), 0, float('inf')], alpha=0.3, colors=['red', 'blue'])  # 绘制决策边界
plt.legend()  # 添加图例
plt.xlabel('X1')  # X 轴标签
plt.ylabel('X2')  # Y 轴标签
plt.title('SVM Classification with Gaussian Kernel')  # 图的标题
plt.show()  # 显示分类结果图像