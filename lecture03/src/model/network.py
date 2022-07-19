import torch
import torch.nn as nn

# use my own block generator
def double_conv(in_channels : int, out_channels : int):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding = 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding = 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

# in pdf
def blockUNet(in_c : int, out_c : int, name : str, size : int = 4, pad : int = 1, transpose : bool = False, bn : bool = True, activation : bool = True, relu : bool = True, dropout : float = 0.1):
    block = nn.Sequential()

    if not transpose:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size = size, padding = pad, stride = 2, bias = True))
    else:
        block.add_module('%s_upsam' % name, nn.Upsample(scale_factor=2, mode = 'bilinear'))
        block.add_module('%s_tconv' % name, nn.Conv2d(in_c, out_c, kernel_size = size-1, padding = pad, stride = 1, bias = True))

    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    
    if dropout > 0:
        block.add_module('%s_dropout' % name, nn.Dropout2d(dropout, inplace = True))
    
    if activation:
        if relu:
            block.add_module('%s_relu' % name, nn.ReLU(inplace = True))
        else:
            block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace = True))

    return block

class DfpNet(nn.Module):
    def __init__(self, channelExponent : int = 6, dropout : float = 0.5):
        super(DfpNet, self).__init__()
        channels = int(2**channelExponent + 0.5)
        self.layer1 = blockUNet(3, channels*1, 'enc_layer1', transpose = False, bn = True, relu = False, dropout = dropout)
        self.layer2 = blockUNet(channels*1, channels*2, 'enc_layer2', transpose = False, bn = True, relu = False, dropout = dropout)
        self.layer3 = blockUNet(channels*2, channels*2, 'enc_layer3', transpose = False, bn = True, relu = False, dropout = dropout)
        self.layer4 = blockUNet(channels*2, channels*4, 'enc_layer4', transpose = False, bn = True, relu = False, dropout = dropout)
        self.layer5 = blockUNet(channels*4, channels*8, 'enc_layer5', transpose = False, bn = True, relu = False, dropout = dropout)
        self.layer6 = blockUNet(channels*8, channels*8, 'enc_layer6', transpose = False, bn = True, relu = False, dropout = dropout, size = 2, pad = 0)
        self.layer7 = blockUNet(channels*8, channels*8, 'enc_layer7', transpose = False, bn = True, relu = False, dropout = dropout, size = 2, pad = 0)
    
        self.dlayer7 = blockUNet(channels*8, channels*8, 'dec_layer7', transpose = True, bn = True, relu = True, dropout = dropout, size = 2, pad = 0)
        self.dlayer6 = blockUNet(channels*16, channels*8, 'dec_layer6', transpose = True, bn = True, relu = True, dropout = dropout, size = 2, pad = 0)
        self.dlayer5 = blockUNet(channels*16, channels*4, 'dec_layer5', transpose = True, bn = True, relu = True, dropout = dropout)
        self.dlayer4 = blockUNet(channels*8, channels*2, 'dec_layer4', transpose = True, bn = True, relu = True, dropout = dropout)
        self.dlayer3 = blockUNet(channels*4, channels*2, 'dec_layer3', transpose = True, bn = True, relu = True, dropout = dropout)
        self.dlayer2 = blockUNet(channels*4, channels*1, 'dec_layer2', transpose = True, bn = True, relu = True, dropout = dropout)
        self.dlayer1 = blockUNet(channels*2, 3, 'dec_layer1', transpose = True, bn = False, activation = False, dropout = dropout)

    def forward(self, x : torch.Tensor)->torch.Tensor:
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)

        dout6 = self.dlayer7(out7)
        dout6_out6 = torch.cat([dout6, out6], dim = 1)
        dout6 = self.dlayer6(dout6_out6)
        dout6_out5 = torch.cat([dout6, out5], dim = 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], dim = 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], dim = 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2 = torch.cat([dout3, out2], dim = 1)
        dout2 = self.dlayer2(dout3_out2)
        dout2_out1 = torch.cat([dout2, out1], dim = 1)
        dout1 = self.dlayer1(dout2_out1)

        return dout1
    
    def weights_init(self)->None:
        for idx, layer in enumerate(self.modules):
            nn.init.uniform_(layer.weight, 0.5,0.5)