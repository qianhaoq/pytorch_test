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




transfer_model = models.vgg16(pretrained=True)
transfer_model = models.vgg19(pretrained=True)

transfer_model = models.resnet34(pretrained=True)
transfer_model = models.resnet50(pretrained=True)
transfer_model = models.resnet101(pretrained=True)
transfer_model = models.resnet152(pretrained=True)