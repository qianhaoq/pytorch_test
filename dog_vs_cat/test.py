# 用mnist的方法实现一下cifar10的图片分类
import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import os

# Hyper Parameters
# 循环次数
num_epochs = 20
# 每批训练的样本数量
batch_size = 128
# 学习速率
learning_rate = 1e-3

transform = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)

# CIFAR10 Dataset
cifar_dataset = dsets.CIFAR10(
    root='./data/', train=True, transform=transform, download=True)
cifar_test_dataset = dsets.CIFAR10(
    root='./data/', train=False, transform=transform)


mnist_dataset = dsets.MNIST(
    root='.', train=True, transform=transform, download=True)
mnist_test_dataset = dsets.MNIST(
    root='.', train=False, transform=transform)

# pokemon_dataset = (
#     dsets.ImageFolder('./pokemon/'
# )
root_dir = os.getcwd() + '/pokemon/'

pokemon_dataset = {
    x: dsets.ImageFolder(os.path.join(root_dir, x),
                            transform)
    for x in ['train', 'test']
}

data_loader = DataLoader(
    dataset = pokemon_dataset['train'],
    batch_size=1,
    shuffle=False,
    num_workers=2
)

to_pil_image = transforms.ToPILImage()
cnt = 0
for idx, (image, label) in enumerate(data_loader, 1):
    # if cnt >= 30:
    #     break
    print(idx, ":  ",label)
    # function 1
    # print(image)
    # print( "=" * 40)
    # print(image[0])
    # exit()
    # img = to_pil_image(image[0])
    img = to_pil_image(image[0])
    print(img.size)
    print(img.height)
    # img.show()

    # function 2
    # img = image[0]
    # img = img.numpy()
    # img = np.transpose(img, (1,2,0))

    # plt.imshow(img)
    # plt.show()


    cnt += 5