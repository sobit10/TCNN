import torch
import torch.nn as nn
import numpy as np

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, causal=False):
        super(DepthwiseSeparableConv, self).__init__()
        depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels, bias=False)

        pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        if causal:
            self.net = nn.Sequential(depthwise_conv,
                                     Chomp1d(padding),
                                     nn.PReLU(),
                                     nn.BatchNorm1d(in_channels),
                                     pointwise_conv)
        else:
            self.net = nn.Sequential(depthwise_conv,
                                     nn.PReLU(),
                                     nn.BatchNorm1d(in_channels),
                                     pointwise_conv)

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ResBlock, self).__init__()

        self.TCM_net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.PReLU(num_parameters=1),
            nn.BatchNorm1d(num_features=out_channels),
            DepthwiseSeparableConv(in_channels=out_channels, out_channels=in_channels, kernel_size=kernel_size,
                                   stride=1,
                                   padding=(kernel_size - 1) * dilation, dilation=dilation, causal=True)
        )

    def forward(self, input):
        x = self.TCM_net(input)
        return x + input


class TCNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, init_dilation=3, num_layers=6):
        super(TCNN_Block, self).__init__()
        layers = []
        for i in range(num_layers):
            dilation_size = init_dilation ** i

            layers += [ResBlock(in_channels, out_channels,
                                kernel_size, dilation=dilation_size)]

        self.network = nn.Sequential(*layers)
        self.pooling = nn.AdaptiveMaxPool1d(1)  # Temporal pooling

    def forward(self, x):
        x = self.network(x)
        x = self.pooling(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, out_channels[0], kernel_size=1)

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[1], kernel_size=1),
            nn.Conv2d(out_channels[1], out_channels[2], kernel_size=3, padding=1)
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[3], kernel_size=1),
            nn.Conv2d(out_channels[3], out_channels[4], kernel_size=5, padding=2)
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels[5], kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)


class DConv2d_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super(DConv2d_block, self).__init__()
        self.DConv2d = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding,
                               output_padding=output_padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.PReLU()
        )
        self.drop = nn.Dropout(0.2)

    def forward(self, encode, decode):
        encode = self.drop(encode)
        skip_connection = torch.cat((encode, decode), dim=1)
        DConv2d = self.DConv2d(skip_connection)

        return DConv2d


class PROP_TCNN(nn.Module):
    def __init__(self):
        super(PROP_TCNN, self).__init__()
        self.Conv2d_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2)),
            nn.BatchNorm2d(num_features=16),
            nn.PReLU()
        )

        self.Conv2d_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2)),
            nn.BatchNorm2d(num_features=16),
            nn.PReLU()
        )

        self.Conv2d_3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 5), stride=(1, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_features=16),
            nn.PReLU()
        )

        self.Conv2d_4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 5), stride=(1, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_features=32),
            nn.PReLU()
        )

        self.Conv2d_5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 5), stride=(1, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_features=32),
            nn.PReLU()
        )
        self.Conv2d_6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 5), stride=(1, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_features=64),
            nn.PReLU()
        )
        self.Conv2d_7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 5), stride=(1, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_features=64),
            nn.PReLU()
        )

        self.TCNN_Block_1 = TCNN_Block(in_channels=256, out_channels=512, kernel_size=3, init_dilation=2, num_layers=6)
        self.TCNN_Block_2 = TCNN_Block(in_channels=256, out_channels=512, kernel_size=3, init_dilation=2, num_layers=6)
        self.TCNN_Block_3 = TCNN_Block(in_channels=256, out_channels=512, kernel_size=3, init_dilation=2, num_layers=6)

        self.DConv2d_7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 5), stride=(1, 2), padding=(1, 1),
                               output_padding=(0, 0)),
            nn.BatchNorm2d(num_features=64),
            nn.PReLU()
        )
        self.DConv2d_6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(3, 5), stride=(1, 2), padding=(1, 1),
                               output_padding=(0, 0)),
            nn.BatchNorm2d(num_features=32),
            nn.PReLU()
        )
        self.DConv2d_5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 5), stride=(1, 2), padding=(1, 1),
                               output_padding=(0, 0)),
            nn.BatchNorm2d(num_features=32),
            nn.PReLU()
        )
        self.DConv2d_4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(3, 5), stride=(1, 2), padding=(1, 1),
                               output_padding=(0, 0)),
            nn.BatchNorm2d(num_features=16),
            nn.PReLU()
        )
        self.DConv2d_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 5), stride=(1, 2), padding=(1, 1),
                               output_padding=(0, 1)),
            nn.BatchNorm2d(num_features=16),
            nn.PReLU()
        )
        self.DConv2d_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2),
                               output_padding=(0, 1)),
            nn.BatchNorm2d(num_features=16),
            nn.PReLU()
        )
        self.DConv2d_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2),
                               output_padding=(0, 0)),
            nn.BatchNorm2d(num_features=1),
            nn.PReLU()
        )

    def forward(self, input):
        Conv2d_1 = self.Conv2d_1(input.float())
        Conv2d_2 = self.Conv2d_2(Conv2d_1)
        Conv2d_3 = self.Conv2d_3(Conv2d_2)
        Conv2d_4 = self.Conv2d_4(Conv2d_3)
        Conv2d_5 = self.Conv2d_5(Conv2d_4)
        Conv2d_6 = self.Conv2d_6(Conv2d_5)
        Conv2d_7 = self.Conv2d_7(Conv2d_6)

        reshape_1 = Conv2d_7.permute(0, 1, 3, 2)  # [64, 64, 4, 5] (B,C,帧长,帧数)
        batch_size, C, frame_len, frame_num = reshape_1.shape
        reshape_1 = reshape_1.reshape(batch_size, C * frame_len, frame_num)

        TCNN_Block_1 = self.TCNN_Block_1(reshape_1)
        TCNN_Block_2 = self.TCNN_Block_2(TCNN_Block_1)
        TCNN_Block_3 = self.TCNN_Block_3(TCNN_Block_2)

        reshape_2 = TCNN_Block_3.reshape(batch_size, C, frame_len, frame_num)
        reshape_2 = reshape_2.permute(0, 1, 3, 2)

        DConv2d_7 = self.DConv2d_7(torch.cat((Conv2d_7, reshape_2), dim=1))
        DConv2d_6 = self.DConv2d_6(torch.cat((Conv2d_6, DConv2d_7), dim=1))
        DConv2d_5 = self.DConv2d_5(torch.cat((Conv2d_5, DConv2d_6), dim=1))
        DConv2d_4 = self.DConv2d_4(torch.cat((Conv2d_4, DConv2d_5), dim=1))
        DConv2d_3 = self.DConv2d_3(torch.cat((Conv2d_3, DConv2d_4), dim=1))
        DConv2d_2 = self.DConv2d_2(torch.cat((Conv2d_2, DConv2d_3), dim=1))
        DConv2d_1 = self.DConv2d_1(torch.cat((Conv2d_1, DConv2d_2), dim=1))

        return DConv2d_1


if __name__ == "__main__":
    x = torch.randn(64, 1, 5, 320)
    model = PROP_TCNN()
    print(model)
  # output torch.Size([64, 1, 8192])
