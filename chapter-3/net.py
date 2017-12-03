from torch import nn

class simpleNet(nn.Module):
    """
    三层全连接网络
    """
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        """
        输入的维度，第一层网络的神经元个数，第二层神经元个数，
        第三层（输出层）神经元个数
        """
        super(simpleNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)


class Activation_Net(nn.Module):
    """
    激活函数
    """
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        """
        输入的维度，第一层网络的神经元个数，第二层神经元个数，
        第三层（输出层）神经元个数
        """
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1, nn_ReLU(True)))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2, nn_ReLU(True)))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)


class Batch_Net(nn.Module):
    """
    批标准化，加速收敛
    """
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.BatchNorm1d(n_hidden_1), nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.BatchNorm1d(n_hidden_2), nn.ReLU(True)           
        )
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, out_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)