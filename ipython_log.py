# IPython log file

# 导入torch基本包
import torch

# 导入varable变量
from torch.autograd import Variable

# 导入神经网络基本函数
import torch.nn.functional as F

# 导入画图包
import matplotlib.pyplot as plt

# 一维转二维，-1到1中取100个点
# x data (tensor) ,shape = (100,1)
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)

# y = x^2 +噪音
y = x.pow(2) + 0.2*torch.rand(x.size())

# 转换为Variable形式，神经网络只能输入Variable
x, y = Variable(x), Variable(y)

##打印散点图
## x.data 是tensor，x.data.numpy是numpy中的数组形式
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x= self.predict(x)
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)
print (net)
