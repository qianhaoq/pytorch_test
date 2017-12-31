# _author = qh
import os
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import models, transforms, datasets
from torchvision.datasets import ImageFolder
from PIL import Image,ImageFile
from datetime import datetime
# solved error:image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def pil_loader(path):
# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

# 定义数据路径
# root_dir = os.getcwd() + '/data/'
root_dir = os.getcwd() + '/data/'

# raw_dir = root_dir + 'raw/'
train_dir = root_dir + 'train/'
# train_dir = '/home/qh/git/comic_crawler/scrawler/pokemon_data/'

# 验证集图片文件夹
test_dir = root_dir + 'test/'

# 将 （1-rate） * 训练图片作为测试集
rate = 0.9

# 判断训练集文件夹是否存在，不存在则创建
create_dir(train_dir)

# 判断测试集文件夹是否存在，不存在则创建
create_dir(test_dir)

# 预处理，将train的图片分别移到dog和cat下
# train_dog_dir = train_dir + 'dog/'
# create_dir(train_dog_dir)

# train_cat_dir = train_dir + 'cat/'
# create_dir(train_cat_dir)

# test_dog_dir = test_dir + 'dog/'
# create_dir(test_dog_dir)

# test_cat_dir = test_dir + 'cat/'
# create_dir(test_cat_dir)

# img_list = os.listdir(raw_dir)

# # 预处理函数
# def pre_process():
#     # 狗图片列表和猫图片的列表
#     dog_list = []
#     cat_list = []

#     # 存入狗列表和猫列表
#     for img in img_list: 
#         if img.split('.')[0] == 'dog':
#             dog_list.append(img)
#         elif img.split('.')[0] == 'cat':
#             cat_list.append(img)

#     # 根据rate比例分配训练集和测试集
#     # dog
#     for i in range(len(dog_list)):
#         img_path = raw_dir + dog_list[i]
#         if i < len(dog_list) * 0.9:
#             obj_path = train_dog_dir + dog_list[i]
#         else:
#             obj_path = test_dog_dir + dog_list[i]
#         # shutil.copyfile(img_path, obj_path)

#     # cat    
#     for i in range(len(cat_list)):
#         img_path = raw_dir + cat_list[i]
#         if i < len(cat_list) * 0.9:
#             obj_path = train_cat_dir + cat_list[i]
#         else:
#             obj_path = test_cat_dir + cat_list[i]
#         # shutil.copyfile(img_path, obj_path)

# def imshow(inp, title=None):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(99)  # pause a bit so that plots are updated

# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            # 32 * 3 * 256 * 256
            # 输入深度3,输出深度16,卷积核大小3*3,0个像素点的填充
            nn.Conv2d(3, 16, kernel_size=3),  # b, 16, 254, 254
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            # 16 * 254 * 254
            nn.Conv2d(16, 32, kernel_size=3),  # b, 32, 252, 252
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  #b, 32, 126, 126
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),  #b, 64, 124, 124
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),  #b, 128, 122, 122
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=20)  #b, 128, 61, 61
        )

        self.fc = nn.Sequential(
            # nn.Linear(128, 10)
            
            nn.Linear(128 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

        self.avg_pool = nn.AvgPool2d(7)


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
        # x = self.avg_pool(x)
        # print(x)
        # print("="*40)        
        x = x.view(x.size(0), -1)
        # print(x)
        # print("="*40)

        x = self.fc(x)
        # print(x)
        # exit()
        return x

# pre_process()
# 定义transforms
# Transforms are common image transforms.
data_transforms = {
    'train':
    transforms.Compose([
        # Crop the given PIL.Image to random size and aspect ratio
        # 随机裁剪原图并转换到指定的大小
        transforms.RandomSizedCrop(256),
        # 50%概率水平翻转
        transforms.RandomHorizontalFlip(),
        # 将pil图片转化为tensor
        transforms.ToTensor(),
        # 将tensor标准化
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),

    'test':
    transforms.Compose([
        # Crop the given PIL.Image to random size and aspect ratio
        # 随机裁剪原图并转换到指定的大小
        transforms.RandomSizedCrop(256),
        # 50%概率水平翻转
        transforms.RandomHorizontalFlip(),
        # 将pil图片转化为tensor
        transforms.ToTensor(),
        # 将tensor标准化
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}
# print(root_dir)

# define datasets
image_datasets = {
    x: datasets.ImageFolder(os.path.join(root_dir, x),
                            data_transforms[x])
    for x in ['train', 'test']
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
train_class_name = image_datasets['train'].classes
test_class_name = image_datasets['test'].classes
print(train_class_name)
print(test_class_name)

# define dataloader to load images
# batch_size = 32

# # resnet18 
# batch_size = 24


# # resnet34
# batch_size = 24

# # vgg 16
batch_size = 16

# # resnet152
# batch_size = 4

dset_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                               shuffle=True, num_workers=4)
                for x in ['train', 'test']
}

print(dset_loaders['train'])

dataloaders = {
    'train':
    DataLoader(
        train_dir,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4),
    'test':
    DataLoader(
        test_dir,
        batch_size = batch_size,
        shuffle = False,
        num_workers= 4)
}


# get train data size and test data size
data_size = {
    'train': len(dset_loaders['train'].dataset),
    'test': len(dset_loaders['test'].dataset)
}
# print(data_size)
# print(dset_loaders['train'].dataset.__len__())
# print(dset_loaders['test'].dataset.__len__())
# exit()
# check gpu  
use_gpu = torch.cuda.is_available()
# use_gpu = False
# 是否修正全连接层的参数
fix_param = True

# # 显示图片的方法
# inputs,classes = next(iter(dset_loaders['train']))
# out = torchvision.utils.make_grid(inputs)
# imshow(out, title=[train_class_name[x] for x in classes])

# 定义一个提前训练好参数的res18模型
# transfer_model = models.resnet18(pretrained=True)

# 预定义模型
# AlexNet: AlexNet variant from the “One weird trick” paper.
# VGG: VGG-11, VGG-13, VGG-16, VGG-19 (with and without batch normalization)
# ResNet: ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152
# SqueezeNet: SqueezeNet 1.0, and SqueezeNet 1.1

transfer_model = models.resnet18(pretrained=True)

# transfer_model = models.vgg16(pretrained=True)
# transfer_model = models.vgg19(pretrained=True)

# transfer_model = models.resnet34(pretrained=True)
# transfer_model = models.resnet50(pretrained=True)
# transfer_model = models.resnet101(pretrained=True)
# transfer_model = models.resnet152(pretrained=True)


# os._exit(0)

# transfer_model = CNN()

# if fix_param:
#     for param in transfer_model.parameters():
#         param.requires_grad = False

# dim_in = transfer_model.fc.in_features
# transfer_model.fc = nn.Linear(dim_in, 2)
if use_gpu:
    transfer_model = transfer_model.cuda()

# define optimize function and loss function
# if fix_param:
#     optimizer = optim.Adam(transfer_model.fc.parameters(), lr=1e-3)
# else:
#     optimizer = optim.Adam(transfer_model.parameters(), lr=1e-3)

# optimizer = optim.Adam(transfer_model.fc.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(transfer_model.parameters(), lr=1e-3)


criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()
# criterion = nn.BCELoss()
# criterion = nn.MultiLabelMarginLoss()

# start train
num_epoch = 1

# print(dset_loaders['train'])
# for i, data in enumerate(dset_loaders['train'], 1):
#     print(i)
#     print(data)
    
#     break
# exit()

for epoch in range(num_epoch):
    print('{}/{}'.format(epoch + 1, num_epoch))
    print('*' * 10)
    print('Train')
    transfer_model.train()
    running_loss = 0.0
    running_acc = 0.0
    since = time.time()
    for i, data in enumerate(dset_loaders['train'], 1):
        img, label = data
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        img = Variable(img)
        label = Variable(label)
        # print("img == ")
        # print(img)
        # print("label == ")
        # print(label)
        # exit()
        # forward
        out = transfer_model(img)
        # print("out == ")
        # print(out)

        # exit()
        loss = criterion(out, label)
        # print(loss)
        # exit()
        _, pred = torch.max(out, 1)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0] * label.size(0)
        num_correct = torch.sum(pred == label)
        running_acc += num_correct.data[0]
        if i % 100 == 0:
            print('Loss: {:.6f}, Acc: {:.4f}'.format(running_loss / (
                i * batch_size), running_acc / (i * batch_size)))
    running_loss /= data_size['train']
    running_acc /= data_size['train']
    elips_time = time.time() - since
    print('Loss: {:.6f}, Acc: {:.4f}, Time: {:.0f}s'.format(
        running_loss, running_acc, elips_time))
    # print('Validation')
    # transfer_model.eval()
    # num_correct = 0.0
    # total = 0.0
    # eval_loss = 0.0
    # for data in dset_loaders['test']:
    #     img, label = data
    #     img = Variable(img, volatile=True)
    #     label = Variable(label, volatile=True)
    #     if use_gpu:
    #         img = img.cuda()
    #         label = label.cuda()
    #     out = transfer_model(img)
    #     _, pred = torch.max(out.data, 1)
    #     loss = criterion(out, label)
    #     eval_loss += loss.data[0] * label.size(0)
    #     num_correct += (pred.cpu() == label.data.cpu()).sum()
    #     total += label.size(0)
    # print('Loss: {:.6f} Acc: {:.4f}'.format(eval_loss / total, num_correct /
    #                                         total))
    # print()
print('Finish Training!')
print('start test!')
transfer_model.eval()
num_correct = 0.0
total = 0.0
eval_loss = 0.0
for data in dset_loaders['test']:
    img, label = data
    img = Variable(img, volatile=True)
    label = Variable(label, volatile=True)
    if use_gpu:
        img = img.cuda()
        label = label.cuda()
    out = transfer_model(img)
    # print("test out")
    # print(out.data)
    # print("test label")
    # print(label)
    # print("data")
    # print(data)
    _, pred = torch.max(out.data, 1)
    print("图片所属宠物小精灵序号")
    print(pred + 1)
    loss = criterion(out, label)
    eval_loss += loss.data[0] * label.size(0)
    num_correct += (pred.cpu() == label.data.cpu()).sum()
    print(num_correct)
    total += label.size(0)
    print(total)
print('Loss: {:.6f} Acc: {:.4f}'.format(eval_loss / total, num_correct /
                                        total))
print('end')
save_path = os.path.join(root_dir, 'model_save' + '_' +datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
if not os.path.exists(save_path):
    os.mkdir(save_path)
# torch.save(transfer_model.state_dict(), save_path + '/resnet18.pth')
torch.save(transfer_model.state_dict(), save_path + '/resnet152.pth')