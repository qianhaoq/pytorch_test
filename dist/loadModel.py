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
from net import feature_net

# solved error:image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

import warnings
warnings.filterwarnings('ignore')


def pil_loader(path):
# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
    with open(path, 'rb') as img:
        return Image.open(img)
# 定义数据路径
# root_dir = os.getcwd() + '/data/'
# root_dir = os.getcwd() + '/data/'
root_dir = "/mnt/data/"
# root_dir = "/home/qh/test/data/"


train_dir = root_dir + 'train/'
# train_dir = root_dir + 'ttt/'


# train_dir = '/home/qh/git/comic_crawler/scrawler/pokemon_data/'

# 验证集图片文件夹
test_dir = root_dir + 'test/'



# pre_process()
# 定义transforms
# Transforms are common image transforms.
data_transforms = {
    'train':
    transforms.Compose([
        # Crop the given PIL.Image to random size and aspect ratio
        # 随机裁剪原图并转换到指定的大小
        transforms.RandomResizedCrop(256),
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
        transforms.RandomResizedCrop(256),
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
                            data_transforms[x],loader = pil_loader)
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

# transfer_model = models.resnet18(pretrained=True)
transfer_model = feature_net("resnet18")
transfer_model.load_state_dict(torch.load('/root/model/resnet18.pth'))
# load_model = torch.load('/home/qh/data/model_save_2018-01-13 10:33:49/resnet152.pth')
# transfer_model = load_model
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
num_epoch = 100

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
    if epoch % 10 == 0:
        torch.save(transfer_model.state_dict(), '/root/model/resnet18.pth' + str(running_acc))
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
# save_path = os.path.join(root_dir, 'model_save' + '_' +datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
# if not os.path.exists(save_path):
#     os.mkdir(save_path)
# # torch.save(transfer_model.state_dict(), save_path + '/resnet18.pth')
# torch.save(transfer_model.state_dict(), save_path + '/resnet152.pth')
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
# torch.save(transfer_model.state_dict(), save_path + '/resnet18.pth')
torch.save(transfer_model.state_dict(), '/root/model/resnet18.pth')
