# 导入库
 
import torch.nn.functional as F
 
import torch.optim as optim
 
import torch
 
import torch.nn as nn
 
import torch.nn.parallel
 
 
 
import torch.optim
 
import torch.utils.data
 
import torch.utils.data.distributed
 
import torchvision.transforms as transforms
 
import torchvision.datasets as datasets
 
 
 
# 设置超参数
 
BATCH_SIZE = 16
 
EPOCHS = 10
 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
 
 
# 数据预处理
 
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
 
 
 
# 读取数据
 
dataset_train = datasets.ImageFolder('data/train', transform)
 
print(dataset_train.imgs)
 
# 对应文件夹的label
 
print(dataset_train.class_to_idx)
 
dataset_test = datasets.ImageFolder('data/val', transform)
 
# 对应文件夹的label
 
print(dataset_test.class_to_idx)
 
# 导入数据
 
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
 
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)
 
 
 
 
 
# 定义网络
 
class ConvNet(nn.Module):
 
    def __init__(self):
 
        super(ConvNet, self).__init__()
 
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
 
        self.max_pool1 = nn.MaxPool2d(2)
 
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
 
        self.max_pool2 = nn.MaxPool2d(2)
 
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
 
        #self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
 
        #self.max_pool3 = nn.MaxPool2d(2)
 
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)

        self.max_pool3 = nn.MaxPool2d(2)
 
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
 
        self.max_pool4 = nn.MaxPool2d(2)
 
        self.fc1 = nn.Linear(25088, 512)
 
        self.fc2 = nn.Linear(512, 1)

 
    def forward(self, x):
 
        in_size = x.size(0)
 
        x = self.conv1(x)
 
        x = F.relu(x)
 
        x = self.max_pool1(x)
 
        x = self.conv2(x)
 
        x = F.relu(x)
 
        x = self.max_pool2(x)
 
        x = self.conv3(x)
 
        x = F.relu(x)
 
        #x = self.conv4(x)
 
        #x = F.relu(x)
 
        #x = self.max_pool3(x)
 
        x = self.conv5(x)
 
        x = F.relu(x)

        x = self.max_pool3(x)
 
        x = self.conv6(x)
 
        x = F.relu(x)
 
        x = self.max_pool4(x)
 
        # 展开
 
        x = x.view(in_size, -1)
 
        x = self.fc1(x)
 
        x = F.relu(x)
 
        x = self.fc2(x)
 
        x = torch.sigmoid(x)
 
        return x
 
 
 
 
 
modellr = 1e-4
 
# 实例化模型并且移动到GPU
 
model = ConvNet().to(DEVICE)

#model = torch.load('model.pth')
print(model)
 
# 选择简单暴力的Adam优化器，学习率调低
 
optimizer = optim.Adam(model.parameters(), lr=modellr)
 
 
 
 
 
def adjust_learning_rate(optimizer, epoch):
 
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
 
    modellrnew = modellr * (0.1 ** (epoch // 5))
 
    print("lr:",modellrnew)
 
    for param_group in optimizer.param_groups:
 
        param_group['lr'] = modellrnew
 
 
 
 
 
# 定义训练过程
 
def train(model, device, train_loader, optimizer, epoch):
 
    model.train()
 
    for batch_idx, (data, target) in enumerate(train_loader):
 
        data, target = data.to(device), target.to(device).float().unsqueeze(1)
 
        optimizer.zero_grad()
 
        output = model(data)
 
        # print(output)
 
        loss = F.binary_cross_entropy(output, target)
 
        loss.backward()
 
        optimizer.step()
 
        if (batch_idx + 1) % 10 == 0:
 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
 
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
 
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))
 
 
 
 
 
# 定义测试过程
 
def val(model, device, test_loader):
 
    model.eval()
 
    test_loss = 0
 
    correct = 0
 
    with torch.no_grad():
 
        for data, target in test_loader:
 
            data, target = data.to(device), target.to(device).float().unsqueeze(1)
 
            output = model(data)
 
            # print(output)
 
            test_loss += F.binary_cross_entropy(output, target, reduction='mean').item()  # 将一批的损失相加
 
            pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in output]).to(device)
 
            correct += pred.eq(target.long()).sum().item()
 
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
 
            test_loss, correct, len(test_loader.dataset),
 
            100. * correct / len(test_loader.dataset)))
 
 
 
 
 
# 训练
 
for epoch in range(1, EPOCHS + 1):
 
    adjust_learning_rate(optimizer, epoch)
 
    train(model, DEVICE, train_loader, optimizer, epoch)
 
    val(model, DEVICE, test_loader)
 
torch.save(model, 'model3.pth')