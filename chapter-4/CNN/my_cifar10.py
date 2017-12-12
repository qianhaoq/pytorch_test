# 用mnist的方法实现一下cifar10的图片分类
import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
import time

# Hyper Parameters
# 循环次数
num_epochs = 20
# 每批训练的样本数量
batch_size = 128
# 学习速率
learning_rate = 1e-3

img_transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

# CIFAR10 Dataset
cifar_dataset = dsets.CIFAR10(
    root='./data/', train=True, transform=img_transforms, download=True)
cifar_test_dataset = dsets.CIFAR10(
    root='./data/', train=False, transform=img_transforms)

mnist_dataset = dsets.MNIST(
    root='./data/', train=True, transform=img_transforms, download=True)
mnist_test_dataset = dsets.MNIST(
    root='./data/', train=False, transform=img_transforms)

# Data Loader (Input Pipeline)
# shuffle表示是否需要随机取样本
cifar_loader = DataLoader(
    dataset=cifar_dataset, batch_size=batch_size, shuffle=True)
cifar_test_loader = DataLoader(
    dataset=cifar_test_dataset, batch_size=batch_size, shuffle=False)

mnist_loader = DataLoader(
    dataset=mnist_dataset, batch_size=batch_size, shuffle=True)
mnist_test_loader = DataLoader(
    dataset=mnist_test_dataset, batch_size=batch_size, shuffle=False)


# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            # 输入深度3,输出深度16,卷积核大小3*3,0个像素点的填充
            nn.Conv2d(3, 16, kernel_size=3),  # b, 16, 30, 30
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),  # b, 32, 28, 28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  #b, 32, 14, 14
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),  #b, 64, 12, 12
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),  #b, 128, 10, 10
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  #b, 128, 5, 5
        )

        self.fc = nn.Sequential(
            # nn.Linear(128, 10)
            
            nn.Linear(128 * 5 * 5, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

        self.avg_pool = nn.AvgPool2d(4)


    def forward(self, x):
        # print("before " + "="*40)
        # print(x)
        # print("layer1 " + "="*40)
        # x = self.layer1(x)
        # print(x)
        # print("layer2 " + "="*40)
        # x = self.layer2(x)
        # print(x)
        # print("layer3 " + "="*40)
        # x = self.layer3(x)
        # print(x)
        # print("layer4 " + "="*40)
        # x = self.layer4(x)
        # print(x)
        # print("avg_pool " + "="*40)
        # x = self.avg_pool(x)
        # print(x)
        # print("x.view " + "="*40)
        # x = x.view(x.size(0), -1)
        # print(x)
        # print("="*40)
        # exit()
        # x = self.fc(x)
        # return x

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        print(x)
        print("="*40)  
        # x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        print(x)
        exit()
        return x


# 记录程序运行时间
start = time.time()
cnn = CNN()
# cnn = AlexNet()
if torch.cuda.is_available():
    cnn.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    
    # for i, (images, labels) in enumerate(cifar_loader):
    #     # 128*3*32*32
    #     print(images)
    #     1 
    # print("="*50)
    for i, (images, labels) in enumerate(cifar_loader):
        # cifar 128*3*32*32
        # print(images)
        if torch.cuda.is_available():
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
        else:
            images = Variable(images)
            labels = Variable(labels)
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)

        # print("label == ")
        # print(labels)
        # print("out == ")
        # print(outputs)
        # exit()
        _, correct_label = torch.max(outputs, 1)
        correct_num = (correct_label == labels).sum()
        acc = correct_num.data[0] / labels.size(0)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, Acc: %.4f' %
                  (epoch + 1, num_epochs, i + 1,
                   len(cifar_loader), loss.data[0], acc))
    
    # for i, (images, labels) in enumerate(train_loader):

        # if torch.cuda.is_available():
        #     images = Variable(images).cuda()
        #     labels = Variable(labels).cuda()
        # else:
        #     images = Variable(images)
        #     labels = Variable(labels)
        # # Forward + Backward + Optimize
        # optimizer.zero_grad()
        # outputs = cnn(images)
        # _, correct_label = torch.max(outputs, 1)
        # correct_num = (correct_label == labels).sum()
        # acc = correct_num.data[0] / labels.size(0)
        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()

        # if (i + 1) % 100 == 0:
        #     print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, Acc: %.4f' %
        #           (epoch + 1, num_epochs, i + 1,
        #            len(train_dataset) // batch_size, loss.data[0], acc))

print("over")

# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in cifar_test_loader:
    if torch.cuda.is_available:
        images = Variable(images.cuda())
    else:
        images = Variable(images)
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()
    
print('Accuracy of the model on the test images: {:.2f} %%'.format(
    100 * correct / total))
# print('Test Accuracy of the model on the 10000 test images: %.4f ' %
#       (correct / total))

# Save the Trained Model
# torch.save(cnn.state_dict(), 'cnn.pth')
elapsed = (time.time() - start)
print(("Time used:",elapsed))