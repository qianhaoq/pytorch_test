# _author = qh
import sys
import os
import shutil
import time
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
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
import torch.nn.functional as F
from resnet import resnet18, resnet34, resnet50, resnet101, resnet152
# vgg out of memory
from vgg import vgg16, vgg19, vgg11
from alexnet import alexnet


para_list = ['vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
para_dict = {}
para_dict['resnet18'] = resnet18()
para_dict['resnet34'] = resnet34()
para_dict['resnet50'] = resnet50()
para_dict['resnet101'] = resnet101()
para_dict['resnet152'] = resnet152()
para_dict['vgg16'] = vgg16()
para_dict['vgg19'] = vgg19()
# solved error:image file is truncatedmpl.use('Agg')  
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

model_name = sys.argv[1]


# root_dir = "/home/qh/tmp_data"
root_dir = "/root/tmp_data"
# root_dir = "/home/qh/test/data/"

def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)

class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block, self).__init__()
        self.same_shape = same_shape
        stride=1 if self.same_shape else 2
        
        self.conv1 = conv3x3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)
        
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), True)
        
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(x+out, True)



train_dir = root_dir + 'train/'


# 验证集图片文件夹
test_dir = root_dir + 'test/'


# create_dir('/root/model/')
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
batch_size = 6

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



LR = 1e-3

# model_SGD = vgg16().cuda()
# model_Momentum = vgg16().cuda()
# model_RMSprop = vgg16().cuda()
model_resnet18 = resnet18().cuda()
model_resnet34 = resnet34().cuda()
model_resnet50 = resnet50().cuda()

# model_Adamax = vgg16().cuda()
# model_Adamax = vgg16().cuda()root_dir = "/home/qh/tmp_data"

# opt_SGD         = torch.optim.SGD(model_SGD.parameters(), lr=LR)
# opt_Momentum    = torch.optim.SGD(model_Momentum.parameters(), lr=LR, momentum=0.8)
# opt_RMSprop     = torch.optim.RMSprop(model_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_resnet18     = torch.optim.Adam(model_resnet18.parameters(), lr=LR, betas=(0.9, 0.99))
opt_resnet34     = torch.optim.Adam(model_resnet34.parameters(), lr=LR, betas=(0.9, 0.99))
opt_resnet50     = torch.optim.Adam(model_resnet50.parameters(), lr=LR, betas=(0.9, 0.99))

# opt_Adamax = torch.optim.Adamax(model_Adamax.parameters(), lr=LR)

# optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam, opt_Adamax]
optimizers = [opt_resnet18, opt_resnet34, opt_resnet50]


# opt_name = ['SGD', 'Momentum', 'RMSprop', 'Adam', 'Adamax']
opt_name = ['resnet18', 'resnet34', 'resnet50']

# models_name = [model_SGD, model_Momentum, model_RMSprop, model_Adam, model_Adamax]
models_name = [model_resnet18, model_resnet34, model_resnet50]

# loss_func = torch.nn.MSELoss()

criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()
# criterion = nn.BCELoss()
# criterion = nn.MultiLabelMarginLoss()

# start train
num_epoch = 20
if len(sys.argv) > 2 :
    num_epoch = int(sys.argv[2])
print('total epoch = ' + str(num_epoch))
# print(dset_loaders['train'])
# for i, data in enumerate(dset_loaders['train'], 1):
#     print(i)
#     print(data)
    
#     break
# exit()


# lines = []
acc_plot = []
loss_plot = []
for idx, optimizer in enumerate(optimizers):
    # transfer_model = feature_net(model_name)
    # if use_gpu:  
    #     transfer_model = transfer_model.cuda()

    acc_list = []
    acc_list.append(0)
    loss_list = []
    loss_list.append(6)
    for epoch in range(num_epoch):
        print('{}/{}'.format(epoch + 1, num_epoch))
        print('*' * 10)
        print('Train')
        models_name[idx].train()
        # transfer_model.train()
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
            # print("out == ")
            # print(transfer_model(img))        
            # out = transfer_model(img)
            out = models_name[idx](img)
            # print("out == ")
            # print(out)

            # exit()
            # out = out.float()
            # label = label.float()
            loss = criterion(out, label)
            # print(loss)
            # exit()
            _, pred = torch.max(out, 1)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("label")
            # print(label+1)
            # print("pred")
            # print(pred+1)
            running_loss += loss.data[0] * label.size(0)
            num_correct = torch.sum(pred == label)
            # print(running_acc)
            running_acc += float(num_correct.data[0])
            # if i % 100 == 0:
            #     print('Loss: {:.6f}, Acc: {:.4f}'.format(running_loss / (
            #         i * batch_size), running_acc / (i * batch_size)))
            #     torch.save(transfer_model.state_dict(), '/home/qh/model/' + model_name + '.pth')

        running_loss /= data_size['train']
        running_acc /= data_size['train']
        loss_list.append(running_loss)
        acc_list.append(running_acc)
        elips_time = time.time() - since
        print('Loss: {:.6f}, Acc: {:.4f}, Time: {:.0f}s'.format(
            running_loss, running_acc, elips_time))
        # if epoch % 10 == 0:
        #     torch.save(transfer_model.state_dict(), '/root/model/resnet18.pth' + str(running_acc))
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
    acc_plot.append(acc_list)
    loss_plot.append(loss_list)
    # x = [x for x in range(0, num_epoch+1)]
    # y = loss_list
    # print(x)
    # print(y)
    # line, = plt.plot(x, y,'',label=opt_name[idx])
    # lines.append(line)
# print(loss_plot)
# print(acc_plot)
# exit()
x = [x for x in range(0, num_epoch+1)]

# fig1 = plt.figure("不同模型与损失值的关系")
# ax1 = fig1.add_subplot(111)
for e in range(len(optimizers)):
    plt.plot(x, loss_plot[e], label=opt_name[e])
plt.title("不同模型与损失值的关系")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig("不同模型与损失值的关系.png")  
plt. close(0)
# fig1.plot(x, y)

# fig2 = plt.figure("不同模型与准确率的关系")
# ax2 = fig2.add_subplot(111)
for e in range(len(optimizers)):
    plt.plot(x, acc_plot[e], label=opt_name[e])
plt.title("不同模型与准确率的关系")
plt.ylabel('Acc')
plt.xlabel('epoch')
plt.legend()
plt.savefig("不同模型与准确率的关系.png")  
plt. close(0)
# plt.show()

# ax2 = fig2.add_subplot(111)
# ax2.plot(x, y, label='kms')

# # print(lines)
# # print(opt_name)
# plt.xlabel("epoch")
# plt.ylabel("Loss")
# plt.legend([lines[0],lines[1],lines[2],lines[3],lines[4]], [opt_name[0],opt_name[1],opt_name[2],opt_name[3],opt_name[4]], loc=1)
# plt.show()

exit()
# save_path = os.path.join(root_dir, 'model_save' + '_' +datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
# if not os.path.exists(save_path):
#     os.mkdir(save_path)
# # torch.save(transfer_model.state_dict(), save_path + '/resnet18.pth')
# torch.save(transfer_model.state_dict(), save_path + '/resnet152.pth')
print('start test!')
transfer_model.eval()
num_correct = 0
total = 0
eval_loss = 0.0
for data in dset_loaders['test']:
    img, label = data
    if use_gpu:
        img = img.cuda()
        label = label.cuda()
    # img = Variable(img)
    # label = Variable(label)
    img = Variable(img, volatile=True)
    label = Variable(label, volatile=True)


    out = transfer_model(img)
    # print("test out")
    # print(out.data)
    # print("test label")
    # print(label)
    # print("data")
    # print(data)
    _, pred = torch.max(out.data, 1)
    # print("原始标签")
    # print(label+1)
    # print("预测标签")
    # print(pred+1)
    loss = criterion(out, label)
    eval_loss += loss.data[0] * label.size(0)
    # num_correct = torch.sum(pred == label)
    num_correct += (pred.cpu() == label.data.cpu()).sum()
    print(num_correct)
    total += label.size(0)
    print(total)

print('Loss: {:.6f} Acc: {:.4f}'.format(eval_loss / total, num_correct /
                                        total))
print('end')
# save_path = os.path.join(root_dir, 'model_save' + '_' +datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
# if not os.path.exists(save_path):
#     os.mkdir(save_path)

torch.save(transfer_model.state_dict(),  '/home/qh/model/' + model_name + '.pth')
# torch.save(transfer_model.state_dict(), '/root/model/' + model_name + '.pth' + +datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))