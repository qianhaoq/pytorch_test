# _author = qh
import os
import torch
from torchvision import models, transforms
from torch import optim, nn
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import shutil

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# 定义数据路径
root_dir = os.getcwd() + '/data/'
raw_dir = root_dir + 'raw/'
train_dir = root_dir + 'train/'
# 验证集图片文件夹
test_dir = root_dir + 'test/'

# 将 （1-rate） * 训练图片作为测试集
rate = 0.9

# 判断训练集文件夹是否存在，不存在则创建
create_dir(train_dir)

# 判断测试集文件夹是否存在，不存在则创建
create_dir(test_dir)

# 预处理，将train的图片分别移到dog和cat下
train_dog_dir = train_dir + 'dog/'
create_dir(train_dog_dir)

train_cat_dir = train_dir + 'cat/'
create_dir(train_cat_dir)

test_dog_dir = test_dir + 'dog/'
create_dir(test_dog_dir)

test_cat_dir = test_dir + 'cat/'
create_dir(test_cat_dir)

img_list = os.listdir(raw_dir)

# 预处理函数
def pre_process():
    # 狗图片列表和猫图片的列表
    dog_list = []
    cat_list = []

    # 存入狗列表和猫列表
    for img in img_list: 
        if img.split('.')[0] == 'dog':
            dog_list.append(img)
        elif img.split('.')[0] == 'cat':
            cat_list.append(img)

    # 根据rate比例分配训练集和测试集
    # dog
    for i in range(len(dog_list)):
        img_path = raw_dir + dog_list[i]
        if i < len(dog_list) * 0.9:
            obj_path = train_dog_dir + dog_list[i]
        else:
            obj_path = test_dog_dir + dog_list[i]
        shutil.copyfile(img_path, obj_path)

    # cat    
    for i in range(len(cat_list)):
        img_path = raw_dir + cat_list[i]
        if i < len(cat_list) * 0.9:
            obj_path = train_cat_dir + cat_list[i]
        else:
            obj_path = test_cat_dir + cat_list[i]
        shutil.copyfile(img_path, obj_path)


pre_process()