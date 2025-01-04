import numpy as np  # 导入numpy库，用于数值计算
import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘图

# 设置高斯分布的参数：均值向量mean和协方差矩阵cov
mean = np.array([2, 1])  # 均值向量，表示数据的中心位置
cov = np.array([[1, -0.5], [0.5, 2]])  # 协方差矩阵，控制数据的散布和形状

# 使用numpy的random.multivariate_normal生成符合给定均值和协方差的多元正态分布的30个点
np.random.seed(0)  # 设置随机种子，保证每次生成的随机数相同
data = np.random.multivariate_normal(mean, cov, 30)  # 生成30个符合上述分布的样本点

# 分别提取x和y坐标
x = data[:, 0].reshape(-1, 1)  # 提取x坐标，并将其转换为n行1列的矩阵
y = data[:, 1]  # 提取y坐标，y是一个一维数组

# 绘制生成的数据点的散点图
plt.scatter(x, y, color='blue', label="Generated Points")  # 绘制蓝色散点图，并标注为"Generated Points"
plt.title('Scatter plot of generated points')  # 设置图像标题
plt.xlabel('x')  # 设置x轴标签
plt.ylabel('y')  # 设置y轴标签
plt.legend()  # 显示图例
plt.show()  # 显示图像

# 线性回归的闭式解
# 构建特征矩阵X_b，第一列为全1（偏置项），第二列为x
X_b = np.c_[np.ones((30, 1)), x]  # np.c_ 将两列数据合并成矩阵，第一列全为1
# 根据正规方程计算最优参数theta_best
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)  # (X.T * X)^(-1) * X.T * y

# 梯度下降法求解线性回归
# 定义梯度下降优化函数
def gradient_descent(X, y, learning_rate=0.1, n_iterations=1000):
    m = len(X)  # 样本数
    theta = np.random.randn(2, 1)  # 随机初始化theta，2行1列矩阵
    for iteration in range(n_iterations):  # 迭代n_iterations次
        gradients = 2 / m * X.T.dot(X.dot(theta) - y.reshape(-1, 1))  # 计算梯度
        theta = theta - learning_rate * gradients  # 更新theta
    return theta  # 返回优化后的theta

# 使用梯度下降法计算最优参数
theta_gd = gradient_descent(X_b, y)

# 使用闭式解和梯度下降法进行预测
y_pred_closed_form = X_b.dot(theta_best)  # 用闭式解的theta_best进行预测
y_pred_gd = X_b.dot(theta_gd)  # 用梯度下降得到的theta_gd进行预测

# 绘制闭式解的线性回归结果
plt.scatter(x, y, color='blue', label="Generated Points")  # 使用scatter()函数绘制原始生成的散点数据，颜色为蓝色，标签为"Generated Points"
plt.plot(x, y_pred_closed_form, color='red', label="Closed-form solution")  # 使用plot()函数绘制闭式解计算得到的线性回归直线，颜色为红色，标签为"Closed-form solution"
plt.title('Linear Regression: Closed-form solution')  # 设置图表的标题为"Linear Regression: Closed-form solution"
plt.xlabel('x')  # 设置x轴的标签为'x'
plt.ylabel('y')  # 设置y轴的标签为'y'
plt.legend()  # 调用legend()函数在图表中显示图例，使得不同曲线对应的标签能出现在图中
plt.show()  # 使用show()函数显示该图像。此时会弹出一个窗口或在脚本中显示图像

# 绘制梯度下降的线性回归结果
plt.scatter(x, y, color='blue', label="Generated Points")  # 再次使用scatter()函数绘制原始生成的散点数据，保持和上一个图一致，颜色为蓝色，标签为"Generated Points"
plt.plot(x, y_pred_gd, color='green', label="Gradient Descent")  # 使用plot()函数绘制通过梯度下降法得到的线性回归直线，颜色为绿色，标签为"Gradient Descent"
plt.title('Linear Regression: Gradient Descent')  # 设置图表的标题为"Linear Regression: Gradient Descent"
plt.xlabel('x')  # 设置x轴的标签为'x'
plt.ylabel('y')  # 设置y轴的标签为'y'
plt.legend()  # 调用legend()函数在图表中显示图例，标记原始点与梯度下降法得到的回归直线
plt.show()  # 使用show()函数显示此图像，同样会弹出一个窗口或在脚本中显示图像

# 输出闭式解和梯度下降法得到的参数
print(theta_best, theta_gd.ravel())  # 输出闭式解和梯度下降法得到的参数，ravel()将矩阵拉平成一维数组

# 梯度下降法并记录每次迭代的损失
def gradient_descent_with_loss(X, y, learning_rate=0.1, n_iterations=1000):
    m = len(X)  # 样本数
    theta = np.random.randn(2, 1)  # 随机初始化theta
    loss_history = []  # 用于记录每次迭代的损失
    for iteration in range(n_iterations):  # 进行n_iterations次迭代
        gradients = 2 / m * X.T.dot(X.dot(theta) - y.reshape(-1, 1))  # 计算梯度
        theta = theta - learning_rate * gradients  # 更新theta
        loss = np.mean((X.dot(theta) - y.reshape(-1, 1)) ** 2)  # 计算当前损失 (均方误差)
        loss_history.append(loss)  # 记录当前损失
    return theta, loss_history  # 返回优化后的theta和损失历史

# 调用梯度下降并记录损失
theta_gd, loss_history = gradient_descent_with_loss(X_b, y)

# 可视化损失函数随迭代次数的变化
plt.plot(range(len(loss_history)), loss_history, color='orange')  # 绘制损失随迭代次数的变化曲线
plt.title('Loss Function vs. Iterations')  # 设置图像标题
plt.xlabel('Iterations')  # x轴标签
plt.ylabel('Mean Squared Error (MSE)')  # y轴标签
plt.show()  # 显示图像