import torch
from torchvision import models
import torch.nn as nn
import torch
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import datasets,transforms
import torch.nn.functional as F
import torch.optim as optim
import csv
import os
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

model = torch.load('model.pth')

test_imgs_path = "data/test"

BATCH_SIZE = 16

transforms = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# test_dataset = datasets.ImageFolder(test_imgs_path, transforms)
# print(test_dataset.class_to_idx)

# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

writer = csv.writer(open("submission.csv", mode='w', encoding='utf-8', newline=""))

model.eval()

headerList = ['id', 'lable']
writer.writerow(headerList)

id = 0
# for i, data in enumerate(test_loader):
#     test_imgs, test_labels = data
#     test_imgs = test_imgs.to(device)
#     test_labels = test_labels.to(device)

#     outputs = model(test_imgs)
#     _,predict_label = torch.max(outputs,1)
#     for j in range(predict_label.size(0)):
#         id += 1
#         writer.writerow([str(id), str(predict_label[j].item())])
#     print(100 * (i + 1) / len(test_loader), '%')

for i in range(5000):#1000表示有1000张图片
    m=i+1
    tpath=os.path.join('./data/test/cat_vs_dog-test/'+ str(m)+'.jpg')     #路径(/home/ouc/river/test)+图片名（img_m）
    fopen = Image.open(tpath)
    data=transforms(fopen)#data就是预处理后，可以送入模型进行训练的数据了
    data = torch.unsqueeze(data, 0)
    outputs = model(data)
    _,predict_label = torch.max(outputs,1)
    for j in range(predict_label.size(0)):
        id += 1
        writer.writerow([str(id), str(predict_label[j].item())])
    print(100 * (i + 1) / 5000, '%')