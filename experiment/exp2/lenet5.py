# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# import numpy as np

# # 网络定义部分
# class LeNet5(nn.Module):
#     def __init__(self):
#         super(LeNet5, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # 输入为1个通道，输出6个特征图，卷积核大小5x5
#         self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # 池化层使用平均池化
#         self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 输入为6个特征图，输出16个特征图，卷积核大小5x5
#         self.conv3 = nn.Conv2d(16, 120, kernel_size=5)  # 输入为16个特征图，输出120个特征图，卷积核大小5x5
#         self.fc1 = nn.Linear(120, 84)  # 全连接层，输入120节点，输出84节点
#         self.fc2 = nn.Linear(84, 10)  # 全连接层，输入84节点，输出10节点

#     def forward(self, x):
#         x = F.relu(self.conv1(x))  # C1层
#         x = self.pool(x)  # S2层
#         x = F.relu(self.conv2(x))  # C3层
#         x = self.pool(x)  # S4层
#         x = F.relu(self.conv3(x))  # C5层
#         x = x.view(-1, 120)  # 展平
#         x = F.relu(self.fc1(x))  # F6层
#         x = F.softmax(self.fc2(x), dim=1)  # 输出层使用softmax激活函数
#         return x

# # 数据集和数据加载
# transform = transforms.Compose([
#     transforms.Resize((32, 32)),  # 调整图像大小为32x32
#     transforms.ToTensor(),  # 转换为tensor
#     transforms.Normalize((0.5,), (0.5,))  # 归一化
# ])

# train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# # 训练部分
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = LeNet5().to(device)
# criterion = nn.CrossEntropyLoss()  # 损失函数使用交叉熵损失
# optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

# epochs = 10
# train_losses = []

# for epoch in range(epochs):
#     model.train()
#     running_loss = 0.0
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)

#         optimizer.zero_grad()  # 梯度清零

#         outputs = model(inputs)  # 前向传播
#         loss = criterion(outputs, labels)  # 计算损失
#         loss.backward()  # 反向传播
#         optimizer.step()  # 更新参数

#         running_loss += loss.item()
#     epoch_loss = running_loss / len(train_loader)
#     train_losses.append(epoch_loss)
#     print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')

# # 绘制损失可视化
# plt.figure(figsize=(10, 5))
# plt.plot(range(1, epochs + 1), train_losses, marker='o')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss Over Epochs')
# plt.grid(True)
# plt.show()

# # 验证部分
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for inputs, labels in test_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')

# # 可视化部分预测结果
# dataiter = iter(test_loader)
# images, labels = next(iter(test_loader))
# images, labels = images.to(device), labels.to(device)
# outputs = model(images)
# _, predicted = torch.max(outputs, 1)

# # 显示部分测试图像及其预测结果
# plt.figure(figsize=(12, 12))
# for idx in range(9):
#     plt.subplot(3, 3, idx + 1)
#     img = images[idx].cpu().numpy().squeeze()  # 将图像转换为numpy格式
#     plt.imshow(img, cmap='gray')
#     plt.title(f'Predicted: {predicted[idx].item()}')
#     plt.axis('off')
# plt.show()

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块
import torch.nn.functional as F  # 导入函数模块，包含激活函数等
import torchvision  # 导入torchvision库，用于处理图像数据
import torchvision.transforms as transforms  # 导入图像转换模块
from torch.utils.data import DataLoader  # 导入数据加载模块
import matplotlib.pyplot as plt  # 导入matplotlib用于绘图
import numpy as np  # 导入numpy进行数据处理
import os  # 导入os模块，用于文件和目录操作
from datetime import datetime  # 导入datetime模块，用于获取当前时间

# 网络定义部分
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()  # 调用父类的构造函数
        # 第一层卷积，输入通道为1，输出通道为6，卷积核大小为5x5
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # 池化层，使用2x2的平均池化，步长为2
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # 第二层卷积，输入为6个特征图，输出16个特征图，卷积核大小为5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 第三层卷积，输入为16个特征图，输出120个特征图，卷积核大小为5x5
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        # 全连接层，输入120节点，输出84节点
        self.fc1 = nn.Linear(120, 84)
        # 最终输出层，全连接，输入84节点，输出10节点（对应10个类别）
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        # 第一层卷积，激活函数使用ReLU
        x = F.relu(self.conv1(x))  # C1层
        # 第一层池化
        x = self.pool(x)  # S2层
        # 第二层卷积，激活函数使用ReLU
        x = F.relu(self.conv2(x))  # C3层
        # 第二层池化
        x = self.pool(x)  # S4层
        # 第三层卷积，激活函数使用ReLU
        x = F.relu(self.conv3(x))  # C5层
        # 展平操作，将多维张量转换为一维向量
        x = x.view(-1, 120)  # 将多维特征图展平为一维
        # 全连接层，激活函数使用ReLU
        x = F.relu(self.fc1(x))  # F6层
        # 最终输出层，不使用softmax，因为交叉熵损失函数会在内部计算softmax
        x = self.fc2(x)  # 输出层
        return x

# 数据集和数据加载
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整图像大小为32x32
    transforms.ToTensor(),  # 转换为tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化，将数据标准化到[-1, 1]
])

# 加载训练集，MNIST数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# 使用DataLoader加载训练数据，batch_size为64，shuffle=True表示每次迭代打乱数据
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 加载测试集，MNIST数据集
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# 使用DataLoader加载测试数据，batch_size为64，shuffle=True表示每次迭代打乱数据
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# 训练部分
# 设置设备，优先使用GPU，如果有GPU可用则使用GPU，否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 初始化模型，并将模型移动到设备上（GPU或CPU）
model = LeNet5().to(device)
# 定义损失函数，使用交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 定义优化器，使用Adam优化器，学习率为0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 用于保存最优模型的准确率
best_accuracy = 0.0
# 创建保存当前时间的文件夹，用于保存模型和可视化结果
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # 获取当前时间，格式为年月日时分秒
base_dir = os.path.join('results', timestamp)  # 创建基础目录路径
checkpoint_dir = os.path.join(base_dir, 'checkpoints')  # 创建用于保存模型的路径
visualization_dir = os.path.join(base_dir, 'visualizations')  # 创建用于保存可视化结果的路径
os.makedirs(checkpoint_dir, exist_ok=True)  # 创建模型保存目录，如果不存在则创建
os.makedirs(visualization_dir, exist_ok=True)  # 创建可视化保存目录，如果不存在则创建

# 训练参数
epochs = 25  # 训练次数
train_losses = []  # 用于保存每个epoch的训练损失

# 训练循环
for epoch in range(epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0  # 初始化当前epoch的损失为0

    # 遍历所有训练数据
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到GPU或CPU上
        optimizer.zero_grad()  # 梯度清零，防止梯度累加
        outputs = model(inputs)  # 前向传播，计算模型输出
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 使用优化器更新模型参数
        running_loss += loss.item()  # 累加当前batch的损失

    # 计算平均损失并保存
    epoch_loss = running_loss / len(train_loader)  # 计算当前epoch的平均损失
    train_losses.append(epoch_loss)  # 保存当前epoch的损失
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')  # 打印当前epoch的损失

    # 验证部分
    model.eval()  # 设置模型为评估模式
    correct = 0  # 初始化预测正确的样本数为0
    total = 0  # 初始化总样本数为0
    with torch.no_grad():  # 不计算梯度，加快计算速度，节省内存
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到GPU或CPU上
            outputs = model(inputs)  # 前向传播，计算模型输出
            _, predicted = torch.max(outputs.data, 1)  # 获取最大概率的类别
            total += labels.size(0)  # 统计测试样本总数
            correct += (predicted == labels).sum().item()  # 统计预测正确的样本数

    accuracy = 100 * correct / total  # 计算准确率
    print(f'Accuracy after Epoch [{epoch + 1}/{epochs}]: {accuracy:.2f}%')  # 打印当前epoch的准确率

    # 每5个epoch保存一次最优模型
    if (epoch + 1) % 5 == 0:
        if accuracy > best_accuracy:  # 仅当当前准确率优于历史最优时保存模型
            best_accuracy = accuracy  # 更新最优准确率
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_lenet5.pth'))  # 保存模型

        # 可视化验证结果
        val_images, val_labels = next(iter(test_loader))  # 获取一个batch的测试数据
        val_images, val_labels = val_images.to(device), val_labels.to(device)  # 将数据移到GPU或CPU上
        val_outputs = model(val_images)  # 前向传播，计算模型输出
        _, val_predicted = torch.max(val_outputs, 1)  # 获取最大概率的类别

        # 创建保存验证可视化结果的目录
        val_vis_dir = os.path.join(visualization_dir, f'epoch_{epoch + 1}')  # 创建保存验证可视化结果的路径
        os.makedirs(val_vis_dir, exist_ok=True)  # 创建目录，如果不存在则创建

        # 可视化前9张验证图像及其预测结果
        plt.figure(figsize=(12, 12))  # 设置图像大小
        for idx in range(9):  # 遍历前9张图像
            plt.subplot(3, 3, idx + 1)  # 创建3x3的子图
            img = val_images[idx].cpu().numpy().squeeze()  # 将图像转换为numpy格式并去掉通道维度
            plt.imshow(img, cmap='gray')  # 显示图像，使用灰度颜色映射
            plt.title(f'Predicted: {val_predicted[idx].item()}')  # 显示预测类别
            plt.axis('off')  # 关闭坐标轴
        plt.savefig(os.path.join(val_vis_dir, 'validation_visualization.png'))  # 保存可视化图像
        plt.close()  # 关闭图像

# 绘制损失可视化
plt.figure(figsize=(10, 5))  # 设置图像大小
plt.plot(range(1, epochs + 1), train_losses, marker='o')  # 绘制训练损失曲线，使用圆形标记
plt.xlabel('Epoch')  # 设置x轴标签
plt.ylabel('Loss')  # 设置y轴标签
plt.title('Training Loss Over Epochs')  # 设置图像标题
plt.grid(True)  # 显示网格线
plt.show()  # 显示图像

# 最终验证部分，计算模型在测试集上的准确率
model.eval()  # 设置模型为评估模式
correct = 0  # 初始化预测正确的样本数为0
total = 0  # 初始化总样本数为0
with torch.no_grad():  # 不计算梯度，加快计算速度，节省内存
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到GPU或CPU上
        outputs = model(inputs)  # 前向传播，计算模型输出
        _, predicted = torch.max(outputs.data, 1)  # 获取最大概率的类别
        total += labels.size(0)  # 统计测试样本总数
        correct += (predicted == labels).sum().item()  # 统计预测正确的样本数

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')  # 打印最终测试集的准确率

# 可视化部分预测结果
# 获取一个batch的测试数据
dataiter = iter(test_loader)  # 创建测试数据的迭代器
images, labels = next(dataiter)  # 获取一个batch的数据
images, labels = images.to(device), labels.to(device)  # 将数据移到GPU或CPU上
outputs = model(images)  # 前向传播，计算模型输出
_, predicted = torch.max(outputs, 1)  # 获取最大概率的类别

# 显示部分测试图像及其预测结果
plt.figure(figsize=(12, 12))  # 设置图像大小
for idx in range(9):  # 遍历前9张图像
    plt.subplot(3, 3, idx + 1)  # 创建3x3的子图
    img = images[idx].cpu().numpy().squeeze()  # 将图像转换为numpy格式并去掉通道维度
    plt.imshow(img, cmap='gray')  # 显示图像，使用灰度颜色映射
    plt.title(f'Predicted: {predicted[idx].item()}')  # 显示预测类别
    plt.axis('off')  # 关闭坐标轴
plt.show()  # 显示图像