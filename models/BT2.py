import torch.nn
import torch.nn.init
import torch.nn.functional as F
class Classifier(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_classes,
                kernel_size=1,
                bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return x

    def init_params(self):
        torch.nn.init.xavier_normal_(self.conv.weight, gain=1.0)
        
def Func_conv3x3(in_channels, out_channels, stride=1, bias=False):
    return torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias)
class Func_conv1x1(torch.nn.Module): 
    #Func_conv1x1(in_channels, out_channels, stride=1, bias=False):
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super().__init__()
        self.conv2d =  torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=bias)
    def forward(self, x):
        return F.relu(self.conv2d(x))
class Block1(torch.nn.Module):
    """    
    """
    def __init__(self, Cin, Cout, stride):
        super().__init__()
       
        self.conv_3x3 = torch.nn.Conv2d(in_channels=Cin, out_channels=2*Cin,
                              kernel_size=3, stride=stride,padding=1)
        self.conv_3x3_1 = torch.nn.Conv2d(in_channels=2*Cin, out_channels=Cout,
                              kernel_size=3,padding=1)
        self.conv_5x5 = torch.nn.Conv2d(in_channels=Cin, out_channels=2*Cin,
                              kernel_size=5, stride=stride, padding=2)

    def forward(self, x):
        x1 = self.conv_3x3(x)
        x2 = self.conv_5x5(x)
        #print('size_x1' + str(x1.size()))
        #print('size_x2' + str(x2.size()))
        x = F.sigmoid(x1*x2)
        x = self.conv_3x3_1(x)
        return F.relu(x)

class Net(torch.nn.Module):
    def __init__(self,
                 num_classes,
                 init_conv_channels,
                 init_conv_stride,
                 channels,
                 strides,
                 in_channels=3,
                 in_size=(224, 224),
                 use_data_batchnorm=True):
                #,groups=2):
        super().__init__()
        self.use_data_batchnorm = use_data_batchnorm
        self.in_size = in_size

        self.backbone = torch.nn.Sequential()

        # data batchnorm
        if self.use_data_batchnorm:
            self.backbone.add_module("data_bn", torch.nn.BatchNorm2d(num_features=in_channels))

        # init conv
        self.backbone.add_module("init_conv", Func_conv3x3(in_channels=in_channels, out_channels=init_conv_channels, stride=init_conv_stride))

        # stages
        in_channels = init_conv_channels
        for stage_id, stage_channels in enumerate(channels):
            stage = torch.nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = strides[stage_id] if unit_id == 0 else 1
                stage.add_module("unit{}".format(unit_id + 1), Block1(Cin=in_channels, Cout=unit_channels, stride=stride))
                in_channels = unit_channels
            self.backbone.add_module("stage{}".format(stage_id + 1), stage)        
       
        self.final_conv_channels = 1024 #Good M4_2 (62.73 stanford dogs)        
        self.backbone.add_module("final_conv", Func_conv1x1(in_channels=in_channels, out_channels=self.final_conv_channels))
        self.backbone.add_module("global_pool", torch.nn.AdaptiveAvgPool2d(output_size=1))
        in_channels = self.final_conv_channels

        # classifier
        self.classifier = Classifier(in_channels=in_channels, num_classes=num_classes)

        self.init_params()

    def init_params(self):
        # backbone
        for name, module in self.backbone.named_modules():            
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)            
            #Thanh Tuan Add start
            elif isinstance(module, torch.nn.Linear):                
                module.weight.data.normal_(0, 0.01)
                module.bias.data.zero_()
            elif isinstance(module, torch.nn.BatchNorm2d):                
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            #Thanh Tuan Add end
        # classifier
        self.classifier.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

def build_Net(num_classes, cifar=True):
    init_conv_channels = 64
    channels = [[128,128],[256,256], [256], [512], [1024]]

    if cifar:
        in_size = (32, 32)
        init_conv_stride = 1
        strides = [1, 1, 2, 2, 2]
    else:
        in_size = (224, 224)
        init_conv_stride = 2
        strides = [1, 2, 2, 2, 2]

    return Net(num_classes=num_classes,
                       init_conv_channels=init_conv_channels,
                       init_conv_stride=init_conv_stride,
                       channels=channels,
                       strides=strides)