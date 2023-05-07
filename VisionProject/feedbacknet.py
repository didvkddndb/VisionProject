import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision.models import resnet50
from torchvision.models import resnet18
# from mobilenet import mobilenetv3_large
class EventDetector(nn.Module):
    def __init__(self):
        super(EventDetector, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.fc = nn.Linear(2048, 1)
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x