import torch.nn as nn
import torch
from collections import OrderedDict
import torch.optim as optim
import torch.nn.functional as F

# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}

def gap2d(x, keepdims=False):
    out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)
    return out

def conv2d_output_size(input_size, kernel_size, stride=1, padding=0, dilation=1):
    input_size = input_size if isinstance(input_size, tuple) else (input_size, input_size)
    kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    stride = stride if isinstance(stride, tuple) else (stride, stride)
    padding = padding if isinstance(padding, tuple) else (padding, padding)
    dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
    output_height = (input_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    output_width = (input_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    return output_height, output_width

def pool2d_output_size(input_size, kernel_size, stride=None, padding=0, dilation=1):
    input_size = input_size if isinstance(input_size, tuple) else (input_size, input_size)
    kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    stride = stride if isinstance(stride, tuple) else (stride, stride) 
    padding = padding if isinstance(padding, tuple) else (padding, padding)
    dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
    output_height = (input_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    output_width = (input_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    return output_height, output_width
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out
    

cfg_list = {
    'vgg11': [[64, 'M'], [128, 'M'], [256, 256, 'M'], [512, 512, 'M'], [512, 512, 'M']],
    'vgg13': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 'M'],[ 512, 512, 'M'], [512, 512, 'M']],
    'vgg16': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 'M'],[512, 512, 512, 'M'], [512, 512, 512, 'M']],
    'vgg19': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 256, 'M'], [512, 512, 512, 512, 'M'], [512, 512, 512, 512, 'M']],
}
def make_layer(cfgs: list,input_size):
    layers = []
    in_channels = input_size[0]
    oh=input_size[1]
    ow=input_size[2]
    for i,v in enumerate(cfgs):
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            oh,ow=pool2d_output_size(input_size=(oh,ow), kernel_size=2, stride=2, padding=0, dilation=1)
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v),nn.ReLU(True)]
            in_channels = v
            oh,ow=conv2d_output_size(input_size=(oh,ow),kernel_size=3, stride=1, padding=1, dilation=1)         
    return nn.Sequential(*layers),(in_channels,oh,ow)

class Net(nn.Module):
    def __init__(self,input_size=(4,256,256),num_classes=2, model_name='vgg16',init_weights=False,for_cam=True):
        assert model_name in cfg_list, "Warning: model number {} not in cfgs dict!".format(model_name)
        super(Net, self).__init__()
        self.input_size=input_size
        self.num_classes=num_classes
        self.for_cam=for_cam
        cfgs = cfg_list[model_name]
        self.layer1,output_size1=make_layer(cfgs[0],input_size)
        self.layer2,output_size2=make_layer(cfgs[1],output_size1)
        self.layer3,output_size3=make_layer(cfgs[2],output_size2)
        self.layer4,output_size4=make_layer(cfgs[3],output_size3)
        self.layer5,output_size5=make_layer(cfgs[4],output_size4)
        self.layer4_2=VGGBlock(output_size5[0]+output_size4[0], output_size4[0], output_size4[0])
        self.layer3_2=VGGBlock(output_size4[0]+output_size3[0], output_size3[0], output_size3[0])
        self.layer2_2=VGGBlock(output_size3[0]+output_size2[0], output_size2[0], output_size2[0])
        self.layer1_2=VGGBlock(output_size2[0]+output_size1[0], output_size1[0], output_size1[0])
        self.up= nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final=nn.Conv2d(output_size1[0], num_classes, kernel_size=1)
        # print(feature_size)
        self.classifier1=nn.Conv2d(output_size5[0], num_classes, 1, bias=False)
        self.classifier2 = nn.Sequential(
            nn.Linear(output_size5[0]*output_size5[1]*output_size5[2], 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

        if init_weights:
            self._initialize_weights()   

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self,inputs):
        x1=self.layer1(inputs)
        x2=self.layer2(x1)
        x3=self.layer3(x2)
        x4=self.layer4(x3)
        x5=self.layer5(x4)

        x4_2=self.layer4_2(torch.cat([x4,self.up(x5)],1))
        x3_2=self.layer3_2(torch.cat([x3,self.up(x4_2)],1))
        x2_2=self.layer2_2(torch.cat([x2,self.up(x3_2)],1))
        x1_2=self.layer1_2(torch.cat([x1,self.up(x2_2)],1))
        x1_2=self.up(x1_2)
        output=self.final(x1_2)
        return {'out':output}
    
    def trainable_parameters(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        return params
    
    def optimizer(self,config,lr=0.0001):
        return optim.Adam(self.trainable_parameters(), lr=lr)

