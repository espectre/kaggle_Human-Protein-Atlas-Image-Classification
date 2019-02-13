import re
import torch
import torch.nn as nn
import numpy as np
import sys,os
sys.path.append("./")
sys.path.append("../")
from DenseNet import densenet121
import settings
config=settings.config_linux

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class DenseNet121(nn.Module):

    def __init__(self,pretrained=True,num_classes=28,start_first=None):
        print("loading the DenseNet121")
        super(DenseNet121, self).__init__()
        self.base_model=densenet121(pretrained=pretrained)

        if start_first=="start_training":
            for i,param in enumerate(self.base_model.parameters()):
                param.requires_grad=False

        self.S=SeparableConv2d(4,3)
        self.relu=nn.ReLU(inplace=True)
        self.bn=nn.BatchNorm2d(4)

        if config['SIZE']==256:
            self.global_pooling=nn.AvgPool2d(kernel_size=8,stride=1)
        elif config['SIZE']==512:
            self.global_pooling=nn.AvgPool2d(kernel_size=16,stride=1)
        else:
            raise NotImplementedError

        self.drop=nn.Dropout2d(p=0.5)
        self.classifier=nn.Linear(1024,num_classes)


    def forward(self,x):
        x=self.bn(x)#过bn层
        x=self.S(x)#分离卷积降成3通道
        x=self.base_model.features(x)#舍弃DenseNet121最后的全连接层
        x=self.relu(x)#relu激活
        x=self.global_pooling(x)#全局池化
        x=x.view(x.size()[0],-1)#拉平
        x=self.drop(x)#dropout
        x=self.classifier(x)#分类

        return x




def test():
    t=np.random.randn(1,4,512,512)

    t=torch.from_numpy(t)

    net=DenseNet121()
    net.cuda()
    output=net(t.float().cuda())
    loss=FocalLoss2d(size_average=False)
    print(output)
    loss_value=loss(output)




if __name__ == '__main__':
    test()