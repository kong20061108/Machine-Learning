import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import pandas as pd
import torchvision
from torch.utils.tensorboard import SummaryWriter
import time
#加载数据集
train_dataset=torchvision.datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)
test_dataset=torchvision.datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor(),download=True)

print('训练集样本数:',len(train_dataset))
print('测试集样本数:',len(test_dataset))

# 创建数据加载器
train_loader=DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)  #shuffle打乱数据
test_loader=DataLoader(dataset=test_dataset,batch_size=64,shuffle=False)    #batch_size决定了dataloader一次拿几张图片，也间接影响后面反向传播一次处理的张量个数

#设备准备
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

# 查看数据的尺寸大小
for images,labels in train_loader:
    print('images.shape:',images.shape)
    print('labels.shape:',labels.shape)
    break;

# 搭建神经网络
class CNN(nn.Module):
    def __init__(self,num_classes):   #num_classes 是种类的数量
        super(CNN,self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(1,32,3),  #卷积和池化会缩小尺寸，计算公式为：处理后=(原尺寸-kernel_size+2*pedding)/stride + 1
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*5*5,64),
            nn.Linear(64,num_classes)
        )

    def forward(self,x):
        x=self.model(x)  #前向传播 记得要输入x
        return x


#设置参数，训练模型
num_classes=len(train_dataset.classes)
epoch=5
train_cnt=0
test_cnt=0
#实例化
cnn=CNN(num_classes)
cnn=cnn.to(device)
#设置损失函数
loss_fn=nn.CrossEntropyLoss()
#优化器
optimizer=torch.optim.Adam(cnn.parameters(),lr=0.001)

start_time=time.time()
writer=SummaryWriter("./logs_train")

#开始训练
for i in range(epoch):
    print(f"-------第{i+1}轮训练开始---------")
    cnn.train()
    for data in train_loader:
        imgs,targets=data
        imgs=imgs.to(device)
        targets=targets.to(device)
        outputs=cnn(imgs)
        loss=loss_fn(outputs,targets)

        optimizer.zero_grad()  #清空当前梯度
        loss.backward()        #反向传播计算梯度
        optimizer.step()        #更新参数

        train_cnt+=1
        if train_cnt%100==0 :
            end_time=time.time()
            print(f"用时:{end_time-start_time}")
            print(f"训练次数:{train_cnt},Loss:{loss.item()}")
            writer.add_scalar("train_loss",loss.item(),train_cnt)


    total_right=0
    total_loss=0
    cnn.eval()
    #测试步骤开始
    with torch.no_grad():
        for data in test_loader:
            imgs,targets=data
            imgs=imgs.to(device)
            targets=targets.to(device)
            outputs=cnn(imgs)
            loss=loss_fn(outputs,targets)
            total_loss+=loss.item()
            right=(outputs.argmax(1)==targets).sum()      #argmax(1)取出每一行的最大值的引索
            total_right+=right

    #打印测试结果
    print(f"测试集上的总loss：{total_loss}")
    print(f"准确率为：{total_right/len(test_dataset)}")

writer.close()