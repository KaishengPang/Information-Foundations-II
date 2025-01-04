import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 定义一个残差块
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # 第一个3x3卷积层
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)  # 批归一化层

        # 第二个3x3卷积层
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)  # 批归一化层

        # Shortcut连接，用于调整输入输出维度不匹配的情况
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            # 如果stride不是1或输入通道数不等于输出通道数，用1x1卷积层调整维度
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        # 前向传播
        out = torch.relu(self.bn1(self.conv1(x)))  # 通过第一个卷积层和激活函数
        out = self.bn2(self.conv2(out))            # 通过第二个卷积层
        out += self.shortcut(x)                    # 残差连接
        return torch.relu(out)                     # 激活输出

# 定义ResNet网络主体
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64  # 初始通道数

        # 网络的第一个卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)              # 批归一化
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大池化

        # 残差层构建，每一层调用_make_layer函数
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # 全局平均池化层和全连接层
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512, num_classes)

    # 构建残差层
    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        layers.append(block(self.in_planes, planes, stride))  # 第一个残差块
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(block(planes, planes))              # 其他残差块
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet的前向传播过程
        out = torch.relu(self.bn1(self.conv1(x)))   # 初始卷积层
        out = self.maxpool(out)                     # 最大池化层
        out = self.layer1(out)                      # 第一个残差层
        out = self.layer2(out)                      # 第二个残差层
        out = self.layer3(out)                      # 第三个残差层
        out = self.layer4(out)                      # 第四个残差层
        out = self.avgpool(out)                     # 全局平均池化
        out = out.view(out.size(0), -1)             # 展平
        out = self.fc(out)                          # 全连接层
        return out

# 定义ResNet-18
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# 数据加载和预处理
transform = transforms.Compose([
    transforms.Resize(224),                          # 调整图像大小到224x224
    transforms.RandomHorizontalFlip(),               # 随机水平翻转
    transforms.RandomCrop(224, padding=4),           # 随机裁剪并填充4个像素
    transforms.ToTensor(),                           # 转换为张量
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # 归一化
])

# CIFAR-10数据集加载
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = ResNet18().to(device)                          # 加载ResNet18模型
criterion = nn.CrossEntropyLoss()                    # 定义交叉熵损失
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)  # 定义Adam优化器

# 训练和验证
train_losses, train_accuracies = [], []  # 用于记录损失和准确率
best_accuracy = 0

for epoch in range(100):  # 训练100个epoch
    net.train()
    train_loss, correct, total = 0, 0, 0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()                  # 梯度清零
        outputs = net(inputs)                  # 前向传播
        loss = criterion(outputs, targets)     # 计算损失
        loss.backward()                        # 反向传播
        optimizer.step()                       # 更新参数

        train_loss += loss.item()
        _, predicted = outputs.max(1)          # 获取预测结果
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()  # 计算正确预测数
    
    avg_loss = train_loss / total              # 计算平均损失
    accuracy = 100. * correct / total          # 计算准确率
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    print(f'Epoch: {epoch}, Loss: {avg_loss}, Accuracy: {accuracy}%')
    
    # 每五个 epoch 检查一次并保存最佳模型
    if epoch % 5 == 0:
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(net.state_dict(), 'best_model.pth')  # 保存最佳模型
            print(f'New best model saved with accuracy: {best_accuracy}%')

# 可视化损失和准确率
epochs = range(1, 101)
fig, ax1 = plt.subplots()

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.plot(epochs, train_losses, color='tab:blue', label='Loss')  # 绘制损失
ax1.fill_between(epochs, train_losses, color='blue', alpha=0.2)

ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy')
ax2.plot(epochs, train_accuracies, color='tab:orange', label='Accuracy')  # 绘制准确率
ax2.fill_between(epochs, train_accuracies, color='orange', alpha=0.2)

fig.tight_layout()
plt.show()

# 测试模型性能，并从测试集挑选一些样本进行预测与真实标签的对比
net.eval()
correct, total = 0, 0
with torch.no_grad():  # 关闭梯度计算，进行测试
    for i, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 显示部分样本的预测结果和真实标签
        if i == 0:  # 只展示第一个batch
            inputs = inputs.cpu()
            for j in range(10):  # 展示10张图片
                img = np.transpose(inputs[j].numpy(), (1, 2, 0))  # 转置以适应显示
                plt.imshow((img * 0.2470 + 0.4914).clip(0, 1))    # 反归一化
                plt.title(f'True: {targets[j].item()}, Pred: {predicted[j].item()}')
                plt.axis('off')
                plt.show()

print(f'Test Accuracy: {100. * correct / total}%')