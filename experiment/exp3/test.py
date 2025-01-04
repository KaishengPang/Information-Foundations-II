import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random

# 定义一个残差块
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)

# ResNet 主体
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# 确保文件夹存在
output_dir = './show'
os.makedirs(output_dir, exist_ok=True)

# 加载数据集（使用和训练相同的预处理）
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# 加载模型并加载权重
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = ResNet18().to(device)
net.load_state_dict(torch.load('best_model.pth'))
net.eval()

# 获取 CIFAR-10 类别标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 保存预测结果的函数，每次随机选择一个批次
def save_predictions(model, test_loader, classes, output_dir):
    model.eval()
    
    # 随机选择一个批次
    batch_index = random.randint(0, len(test_loader) - 1)
    data_iter = iter(test_loader)
    for _ in range(batch_index):
        images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    # 获取模型预测
    with torch.no_grad():
        outputs = model(images)
        _, predicted = outputs.max(1)
    
    # 保存前100张图片及预测结果
    for i in range(100):
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        img = img * np.array([0.2470, 0.2435, 0.2616]) + np.array([0.4914, 0.4822, 0.4465])  # 反归一化
        img = np.clip(img, 0, 1)  # 确保像素值在合理范围

        plt.figure(figsize=(1.6, 1.6))  # 调整图像大小为 32x32
        plt.imshow(img)
        plt.title(f'True: {classes[labels[i]]}\nPred: {classes[predicted[i]]}')
        plt.axis('off')
        
        # 提高保存质量
        plt.savefig(os.path.join(output_dir, f'image_{i}.png'), dpi=200, bbox_inches='tight', pad_inches=0)
        plt.close()

# 进行保存
save_predictions(net, testloader, classes, output_dir)