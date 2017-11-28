# -*- coding: utf-8 -*
# author: qh
# date: 2017/11/28
# 原版本为python2.7，待改编为python3
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.autograd import Variable


# 定义初始节点
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# numpy.array 转 tensor
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


class LinearRegression(nn.Module):
    """
    建立线性回归模型
    """
    def __init__(self):
        super(LinearRegression, self).__init__()
        # 定义输入输出是1维的
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

# 如果cuda可用则用cuda加速
if torch.cuda.is_available():
    model = LinearRegression().cuda()
else:
    model = LinearRegression()

# 定义损失函数，nn.MSELoss代表平方损失函数
criterion = nn.MSELoss()

# 定义优化函数, SGD代表随机梯度下降 ,lr代表学习速度
# 例子  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  
# 使用SGD（随机梯度下降）优化，学习率为0.001，动量为0.9
# 此处的1e-3代表 0.001，即10的-3次方
optimizer = optim.SGD(model.parameters(), lr = 1e-3)

# 定义总的循环次数
num_epochs = 1000
# 循环
for epoch in range(num_epochs):
    if torch.cuda.is_available():
        inputs = Variable(x_train).cuda()
        target = Variable(y_train).cuda()
    else:
        inputs = Variable(x_train)
        target = Variable(y_train)

    # forward向前反馈
    # 得到网络向前传播的结果Out
    out = model(inputs)
    # 得到损失函数
    loss = criterion(out, target)

    # backward
    # 每次反向传播之前都要归零梯度，不然梯度会累加，造成结果不收敛
    # 归零梯度
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 更新参数
    optimizer.step()

    # 每隔固定时间观察损失函数的值
    if (epoch+1) % 20 == 0:
        print('Epoch[{}/{}], loss:{:.6f}'.format(epoch + 1, num_epochs, loss.data[0]))

# 将模型变成测试模式(有一些层在测试与训练的时候是不一样的,比如Dropout,BatchNormalization) 
model.eval()

predict = model(Variable(x_train))
predict = predict.data.numpy()

# 画图
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
plt.plot(x_train.numpy(), predict, label='Fitting Line')
# 展现
plt.show()