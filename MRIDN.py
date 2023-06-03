import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable

class ConvRelu(nn.Sequential):
    def __init__(self,inchannel,outchannel,kernel_size,padding):
        super(ConvRelu, self).__init__(
            nn.Conv2d(inchannel,outchannel,kernel_size,padding=padding),
            nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )

class Unetup(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(Unetup, self).__init__()
        self.uosampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(inchannel,outchannel,kernel_size=3,padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2,inplace=True)
        self.conv2 = nn.Conv2d(inchannel,inchannel//2,kernel_size=3,padding=1)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv3 = nn.Conv2d(inchannel//2,inchannel//2,kernel_size=3,padding=1)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self,input1,input2):
        input1 = self.uosampling(input1)
        input1 = self.conv1(input1)
        input1 = self.relu1(input1)
        x = torch.cat([input1,input2],dim=1)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        return x


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))
def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
class daf(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(daf, self).__init__()
        self.layernorm = LayerNorm2d(inchannel)
        self.conv1 = nn.Conv2d(inchannel, inchannel, 1)
        self.conv2 = nn.Conv2d(inchannel, inchannel, 3, padding=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.gate = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=inchannel // 2, out_channels=inchannel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.conv3 = nn.Conv2d(inchannel // 2, inchannel, 1)
        self.conv4 = nn.Conv2d(inchannel, outchannel, 1)
    def forward(self,x):

        x1 = self.layernorm(x)
        x1 = self.conv1(x1)
        x2 = self.conv2(x1)
        x3 = x1 + x2
        x3 = self.gate(x3)
        x3 = x3 * self.sca(x3)
        x3 = self.conv3(x3) + x
        x4 = self.conv4(x3)
        return x4


class Unetup_last(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(Unetup_last, self).__init__()
        self.uosampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(inchannel,outchannel,kernel_size=3,padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2,inplace=True)
        self.conv2 = nn.Conv2d(inchannel,inchannel//2,kernel_size=3,padding=1)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv3 = nn.Conv2d(inchannel//2,3,kernel_size=3,padding=1)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)


    def forward(self,input1,input2):
        input1 = self.uosampling(input1)
        input1 = self.conv1(input1)
        input1 = self.relu1(input1)
        x = torch.cat([input1,input2],dim=1)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        return x


class IMDModule(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModule, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = ConvRelu(in_channels, in_channels, 3, padding=1)
        self.c2 = daf(self.remaining_channels, in_channels)
        self.c3 = daf(self.remaining_channels, in_channels)
        self.c4 = daf(self.remaining_channels, self.distilled_channels)
        self.c5 = ConvRelu(in_channels, in_channels, kernel_size=1,padding=0)
        self.cca = CCALayer(self.distilled_channels * 4)

    def forward(self, input):
        out_c1 = input
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.c2(remaining_c1)
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.c3(remaining_c2)
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(remaining_c3)
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out = self.c5(out)
        out = torch.cat([input,out],dim=1)
        return out

class UnetSE(nn.Module):
    def __init__(self,inchannel,outchannel=32):
        super(UnetSE, self).__init__()
        self.features = nn.Sequential(
            ConvRelu(inchannel,outchannel,kernel_size=1,padding=0),
            ConvRelu(outchannel,outchannel,kernel_size=3,padding=1),
            nn.MaxPool2d(2),
            IMDModule(outchannel),
            # ConvRelu(outchannel,outchannel*2,kernel_size=3,padding=1),
            ConvRelu(outchannel*2,outchannel*2,kernel_size=1,padding=0),
            nn.MaxPool2d(2),
            IMDModule(outchannel*2),
            # ConvRelu(outchannel*2,outchannel*4,kernel_size=3,padding=1),
            ConvRelu(outchannel*4,outchannel*4,kernel_size=1,padding=0),
            nn.MaxPool2d(2),
            IMDModule(outchannel*4),
            # ConvRelu(outchannel*4,outchannel*8,kernel_size=3,padding=1),
            ConvRelu(outchannel*8,outchannel*8,kernel_size=1,padding=0),
            nn.MaxPool2d(2),
            IMDModule(outchannel*8),
            # ConvRelu(outchannel*8,outchannel*16,kernel_size=3,padding=1),
            ConvRelu(outchannel*16,outchannel*16,kernel_size=1,padding=0),
            ConvRelu(outchannel*16,outchannel*16,kernel_size=3,padding=1),
            # nn.BatchNorm2d(outchannel*16),
        )

        self.cca = CCALayer(outchannel*16)

        self.concat1 = Unetup(inchannel=outchannel*16,outchannel=outchannel*8)
        self.concat2 = Unetup(inchannel=outchannel*8,outchannel=outchannel*4)
        self.concat3 = Unetup(inchannel=outchannel*4,outchannel=outchannel*2)
        self.concat4 = Unetup_last(inchannel=outchannel*2,outchannel=outchannel)

    def forward(self,x):

        conv1 = self.features[0:2](x)
        conv2 = self.features[2:5](conv1)
        conv3 = self.features[5:8](conv2)
        conv4 = self.features[8:11](conv3)
        conv5 = self.features[11:-1](conv4)
        x = conv5
        x = self.cca(x)

        x = self.concat1(x,conv4)
        x = self.concat2(x,conv3)
        x = self.concat3(x,conv2)
        x = self.concat4(x,conv1)
        x = torch.clamp(x,min=0,max=1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight,gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)


if __name__ == '__main__':
    x = torch.Tensor(1,3,64,64)
    model = UnetSE(inchannel=3,outchannel=32)
    out = model(x)
    print(out.shape)

