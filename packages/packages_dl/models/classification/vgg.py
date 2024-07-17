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
        self.layer1,output_size=make_layer(cfgs[0],input_size)
        self.layer2,output_size=make_layer(cfgs[1],output_size)
        self.layer3,output_size=make_layer(cfgs[2],output_size)
        self.layer4,output_size=make_layer(cfgs[3],output_size)
        self.layer5,output_size=make_layer(cfgs[4],output_size)
        # print(feature_size)
        self.classifier1=nn.Conv2d(output_size[0], num_classes, 1, bias=False)
        self.classifier2 = nn.Sequential(
            nn.Linear(output_size[0]*output_size[1]*output_size[2], 4096),
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
        # input1=inputs[:,0:self.input_channel_list[0]]
        x=self.layer1(inputs)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.layer5(x)

        # print(x.shape)
        # N x 512 x 7 x 7
        if self.for_cam:

            cam = F.conv2d(x, self.classifier1.weight)
            cam=F.relu(cam)
            cam=cam[0]+cam[1].flip(-1)

            x=gap2d(x,keepdims=True)
            # print(x.shape)
            x=self.classifier1(x)
            # print(x.shape)
            x = x.view(-1, self.num_classes)
            return {'cam':cam,'out':x}
        else:
            x = torch.flatten(x, start_dim=1)
            x=self.classifier2(x)
            return {'out':x}

    def trainable_parameters(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        return params
    
    def optimizer(self,config,lr=0.0001):
        return optim.Adam(self.trainable_parameters(), lr=lr)


class CAM(Net):
    def __init__(self,input_size=(4,256,256),num_classes=2, model_name='vgg16',init_weights=False):
        super(CAM, self).__init__(input_size=input_size,num_classes=num_classes,model_name=model_name,init_weights=init_weights)
    def forward(self,x,step=1):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.layer5(x)
        print(x.shape)
        cam = F.conv2d(x, self.classifier1.weight)
        print(cam.shape)
        return cam
