import torch
from torchvision import models
import torch.nn as nn
import torch
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import datasets,transforms
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

#densenet = models.densenet121(weights=None)

densenet = torch.load('model3.pth')

train_imgs_path = "data/train"
valid_imgs_path = "data/val"

EPOCH = 10
BATCH_SIZE = 16
LEARN_RATE = 0.0001

transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1)), #随机裁剪一个area然后再resize
    transforms.RandomHorizontalFlip(), #随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    
train_dataset = datasets.ImageFolder(train_imgs_path, transforms)
valid_dataset = datasets.ImageFolder(valid_imgs_path, transforms)

print(train_dataset.class_to_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

# densenet.classifier = nn.Sequential(
#     nn.Linear(1024, 512),
#     nn.Dropout(0.25),
#     nn.ReLU(),
#     nn.Linear(512, 2)
# )

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(densenet.parameters(), lr=LEARN_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

def train(epoch):
    correct = 0
    total = 0
    losses = 0
    densenet.train() # 设为训练模式
    for i, data in enumerate(train_loader):
        train_imgs, train_labels = data
        train_imgs = train_imgs.to(device)
        train_labels = train_labels.to(device)

        outputs = densenet(train_imgs)
        _, predict_label = torch.max(outputs, 1)

        total += train_labels.size(0)
        correct += (predict_label == train_labels).sum().item()
        loss = criterion(outputs, train_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.item()
        if i % 5 == 0:
            print("train :  epoch =", epoch, " step =", i + 1,
                " {:.1f}%".format(100 * (i + 1) / len(train_loader)), " loss =", loss.item())
    # python的i可以留到循环外
    
    print('\n', "train :  epoch =", epoch,
        " learn rate =" , optimizer.param_groups[0]['lr'],
        " loss =", losses / (i + 1), " accuracy =", correct / total, '\n')


def valid(epoch):
    losses = 0
    correct = 0
    total = 0
    densenet.eval()
    for i, data in enumerate(valid_loader):
        valid_imgs,valid_labels = data
        valid_imgs = valid_imgs.to(device)
        valid_labels = valid_labels.to(device)

        outputs = densenet(valid_imgs)
        loss = criterion(outputs, valid_labels)
        losses += loss.item()
        _,predict_label = torch.max(outputs,1)
        total += valid_labels.size(0)
        correct += (predict_label == valid_labels).sum().item()
        if i % 5 == 0:
            print("validation :  epoch =", epoch, " step =", i + 1,
                " {:.1f}%".format(100 * (i + 1) / len(valid_loader)), " loss =", loss.item())
    scheduler.step(losses / (i + 1))
    print('\n', "validation :  epoch =", epoch, " loss =", 
        losses / (i + 1), " accuracy =", correct / total, '\n')

for epoch in range(1, EPOCH + 1):
    #train(epoch)
    valid(epoch)
    #torch.save(densenet, 'modellast' + str(epoch) + '.pth')
    #print('model saved')