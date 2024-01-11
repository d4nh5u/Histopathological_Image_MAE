import torch
from torch import Tensor
import torch.nn as nn
from torchinfo import summary
from torch.nn import functional as F

from pathlib import Path
import sys
if str(Path().absolute()) not in sys.path:
    sys.path.append(str(Path().absolute()))
    
#DenseNet

class DenseLayer(nn.Sequential):
  """Basic unit of DenseBlock (using bottleneck layer) """
  def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
    super(DenseLayer, self).__init__()
    self.bn1 = nn.BatchNorm2d(num_input_features)
    self.relu1 = nn.ReLU()
    self.conv1 = nn.Conv2d(num_input_features, bn_size*growth_rate, kernel_size=1, stride=1, bias=False)
    self.bn2 = nn.BatchNorm2d(bn_size*growth_rate)
    self.relu2 = nn.ReLU()
    self.conv2 = nn.Conv2d(bn_size*growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
    self.drop_rate = drop_rate

  def forward(self, x):
    output = self.bn1(x)
    output = self.relu1(output)
    output = self.conv1(output)

    output = self.bn2(output)
    output = self.relu2(output)
    output = self.conv2(output)

    if self.drop_rate > 0:
      output = F.dropout(output, p=self.drop_rate)
    return torch.cat([x, output], 1)



class DenseBlock(nn.Sequential):
  def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
    super(DenseBlock, self).__init__()
    for i in range(num_layers):
      if i == 0:
        self.layer = nn.Sequential(DenseLayer(num_input_features+i*growth_rate, growth_rate, bn_size,drop_rate))
      else:
        layer = DenseLayer(num_input_features+i*growth_rate, growth_rate, bn_size,drop_rate)
        self.layer.add_module("denselayer%d" % (i+1), layer)

  def forward(self,input):
    return self.layer(input)


class Transition(nn.Sequential):
  def __init__(self, num_input_feature, num_output_features):
    super(Transition, self).__init__()
    self.bn = nn.BatchNorm2d(num_input_feature)
    self.relu = nn.ReLU()
    self.conv = nn.Conv2d(num_input_feature, num_output_features, kernel_size=1, stride=1, bias=False)
    self.pool = nn.AvgPool2d(2, stride=2)

  def forward(self,input):
    output = self.bn(input)
    output = self.relu(output)
    output = self.conv(output)
    output = self.pool(output)
    return output

class DenseNet(nn.Module):
  def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=224, bn_size=4, compression_rate=0.5, drop_rate=0, num_classes=5):
    super(DenseNet, self).__init__()

    self.features = nn.Sequential(
      #first layer
      nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
      nn.BatchNorm2d(num_init_features),
      nn.ReLU(),
      #second layer
      nn.MaxPool2d(3, stride=2, padding=1)
    )

    # DenseBlock
    num_features = num_init_features
    for i, num_layers in enumerate(block_config):
      block = DenseBlock(num_layers, num_features, bn_size, growth_rate,drop_rate)
      if i == 0:
        self.block_tran = nn.Sequential(block)
      else:
        self.block_tran.add_module("denseblock%d" % (i + 1), block)

      num_features += num_layers*growth_rate
      if i != len(block_config) - 1:
        transition = Transition(num_features, int(num_features*compression_rate))
        self.block_tran.add_module("transition%d" % (i + 1), transition)
        num_features = int(num_features * compression_rate)


    self.tail = nn.Sequential(nn.BatchNorm2d(num_features), nn.ReLU())


    self.classifier = nn.Linear(num_features, num_classes)

    if block_config == (6, 12, 24, 16):
      self.name = "DenseNet121"
    elif block_config == (6, 12, 32, 32):
      self.name = "DenseNet169"


    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1)
      elif isinstance(m, nn.Linear):
        nn.init.constant_(m.bias, 0)

  def forward(self, x):
    features = self.features(x)
    block_output = self.block_tran(features)
    tail_output = self.tail(block_output)
    out = F.avg_pool2d(tail_output, 7, stride=1).view(tail_output.size(0), -1)
    out = self.classifier(out)
    return out

def DenseNet121():
    return DenseNet()

def DenseNet169():
    return DenseNet(block_config=(6, 12, 32, 32))



if __name__ == '__main__':
    project_path = Path().absolute()
    print('Project Path:', project_path)
    
    model = DenseNet121()
    model.to('cpu')
    summary(model, input_size=(1, 3, 224, 224))
    
    model_1 = DenseNet169()
    model_1.to('cpu')
    summary(model_1, input_size=(1, 3, 224, 224))