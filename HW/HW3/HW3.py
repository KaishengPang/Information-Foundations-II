import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保可重复性
np.random.seed(0)

# A类点的参数
mean_A = [0, 0]  # A类点的均值
cov_A = [[1, 0], [0, 1]]  # A类点的协方差矩阵
num_A = 30  # A类点的数量

# B类点的参数
mean_B = [1, 2]  # B类点的均值
cov_B = [[2, 0], [0, 2]]  # B类点的协方差矩阵
num_B = 30  # B类点的数量

# 生成A类和B类的随机点
A_points = np.random.multivariate_normal(mean_A, cov_A, num_A)  # 生成A类点
B_points = np.random.multivariate_normal(mean_B, cov_B, num_B)  # 生成B类点

# 创建标签
A_labels = np.zeros(num_A)  # A类标签为0
B_labels = np.ones(num_B)   # B类标签为1

# 合并数据和标签
X = np.vstack((A_points, B_points))  # 合并A类和B类的点
y = np.concatenate((A_labels, B_labels))  # 合并标签

# 绘制生成的点
plt.scatter(A_points[:, 0], A_points[:, 1], label='Class A', color='blue', marker='o')  # 绘制A类点
plt.scatter(B_points[:, 0], B_points[:, 1], label='Class B', color='red', marker='x')    # 绘制B类点
plt.title('Points Map')  # 设置标题
plt.xlabel('X1')  # 设置x轴标签
plt.ylabel('X2')  # 设置y轴标签
plt.legend()  # 显示图例
plt.grid()  # 显示网格
plt.show()  # 显示绘制的图形

# 定义逻辑回归模型
class LogisticRegressionCustom:
    def __init__(self, learning_rate=0.1, num_iterations=10000):
        self.learning_rate = learning_rate  # 学习率
        self.num_iterations = num_iterations  # 迭代次数
        self.weights = None  # 权重初始化为None
        self.bias = None  # 偏置初始化为None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))  # Sigmoid激活函数

    def compute_loss(self, y_true, y_pred):
        m = len(y_true)
        # 计算交叉熵损失
        loss = - (1/m) * np.sum(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
        return loss

    def fit(self, X, y):
        num_samples, num_features = X.shape  # 获取样本数量和特征数量
        self.weights = np.zeros(num_features)  # 初始化权重为0
        self.bias = 0  # 初始化偏置为0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias  # 计算线性模型
            y_predicted = self.sigmoid(linear_model)  # 应用sigmoid函数得到预测值

            # 计算当前损失
            loss = self.compute_loss(y, y_predicted)
            if _ % 1000 == 0:  # 每1000次迭代输出一次损失
                print(f'Iteration {_}, Loss: {loss:.4f}')

            # 梯度下降
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))  # 计算权重的梯度
            db = (1 / num_samples) * np.sum(y_predicted - y)  # 计算偏置的梯度

            self.weights -= self.learning_rate * dw  # 更新权重
            self.bias -= self.learning_rate * db  # 更新偏置

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias  # 计算线性模型
        y_predicted = self.sigmoid(linear_model)  # 应用sigmoid函数
        class_labels = [1 if i > 0.5 else 0 for i in y_predicted]  # 根据阈值0.5确定类标签
        return np.array(class_labels)  # 返回类标签的数组

# 初始化逻辑回归模型
model = LogisticRegressionCustom()

# 训练模型
model.fit(X, y)

# 预测
y_pred = model.predict(X)

# 计算准确率
accuracy = np.mean(y_pred == y)  # 计算预测的准确率
print(f'Accuracy: {accuracy:.2f}')  # 打印准确率

# 创建网格点以绘制决策边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # 设置x轴的范围
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # 设置y轴的范围
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))  # 创建网格点

# 预测网格点的类别
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])  # 预测网格点的类标签
Z = Z.reshape(xx.shape)  # 将预测结果重新调整为网格的形状

# 绘制决策边界
plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.RdYlBu)  # 绘制决策边界
plt.scatter(A_points[:, 0], A_points[:, 1], label='Class A', color='blue', marker='o')  # 绘制A类点
plt.scatter(B_points[:, 0], B_points[:, 1], label='Class B', color='red', marker='x')  # 绘制B类点
plt.title('Logistic Regression Map')  # 设置标题
plt.xlabel('X1')  # 设置x轴标签
plt.ylabel('X2')  # 设置y轴标签
plt.legend()  # 显示图例
plt.grid()  # 显示网格
plt.show()  # 显示绘制的图形
