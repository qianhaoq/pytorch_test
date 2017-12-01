# -*- coding: utf-8 -*
# author: qh
# date: 2017/11/28
# 多项式回归
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.autograd import Variable


w_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
b_target = torch.FloatTensor([0.9])


def make_features(x):
    """
     build features for a matrix with colums [x, x^2, x^3]
     
     torch.squeeze(input, dim=None, out=None)
     Returns a Tensor with all the dimensions of input of size 1 removed.
     默认是去掉所有维度中维度大小为1的维度并返回
     若指定了dim（从0开始），则判断该维度大小是否为1,若为1则去掉

     torch.unsqueeze(input, dim, out=None)
     Returns a new tensor with a dimension of size one inserted at the specified position.
     在指定位置插入一个1维tensor
    """
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, 4)], 1)

def f(x):
    """
    模拟函数，输入一个x得到一个y
    """
    return x.mm(w_target) + b_target[0]

def plot_function(model):
    """
    画图
    """
    x_data = make_features(torch.arange(-1, 1, 0.1))
    y_data = f(x_data)
    if torch.cuda.is_available():
        y_pred = model(Variable(x_data).cuda())
    x = torch.arange(-1, 1, 0.1).numpy()
    y = y_data.numpy()
    y_p = y_pred.cpu().data.numpy()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, y, 'ro', label='real curve')
    plt.plot(x, y_p, label='fitting curve')
    plt.legend(loc='best')
    plt.show()

# print funciton describe
def poly_desc(w, b):
    des = 'y = {:.2f} + {:.2f}*x + {:.2f}*x^2 + {:.2f}*x^3'.format(
        b[0], w[0], w[1], w[2])
    return des

def get_batch(batch_size = 32):
    """
    随机生成一些数来得到每次的训练集
    """
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    if torch.cuda.is_available():
        return Variable(x).cuda(), Variable(y).cuda()
    else:
        return Variable(x), Variable(y)

class poly_model(nn.Module):
    """
    建立多项式模型
    """
    def __init__(self):
        super(poly_model, self).__init__()
        # 定义输入输出是3*1维的
        self.poly = nn.Linear(3, 1)

    def forward(self, x):
        out = self.poly(x)
        return out

# 如果cuda可用则用cuda加速
# class torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)[source]
# 在模块级别上实现数据并行。
# http://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/#torchnn
if torch.cuda.is_available():
    model = poly_model().cuda()
    # model = poly_model()
    # model = torch.nn.DataParallel(model).cuda()
else:
    model = poly_model()

# 定义损失函数，nn.MSELoss代表平方损失函数
criterion = nn.MSELoss()

# 定义优化函数, SGD代表随机梯度下降 ,lr代表学习速度
# 例子  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  
# 使用SGD（随机梯度下降）优化，学习率为0.001，动量为0.9
# 此处的1e-3代表 0.001，即10的-3次方
optimizer = optim.SGD(model.parameters(), lr = 1e-3)

# 定义总的循环次数
epoch = 0
while True:
    # Get data
    batch_x, batch_y = get_batch()
    # forward pass
    output = model(batch_x)
    loss = criterion(output, batch_y)
    print_loss = loss.data[0]
    optimizer.zero_grad()
    # Backward pass
    loss.backward()
    # 优化参数
    optimizer.step()
    epoch += 1
    if print_loss < 1e-3:
        torch.save(model.state_dict(), './poly.pth')
        break

print('Loss: {:.6f} after {} batches'.format(print_loss, epoch))
print('==> Learned function:\t' + poly_desc(model.poly.weight.data.view(-1),
                                            model.poly.bias.data))
print('==> Actual function:\t' + poly_desc(w_target.view(-1), b_target))
plot_function(model)