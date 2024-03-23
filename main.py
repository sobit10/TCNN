import os
import results
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from others.root import *
from scipy.io import wavfile
from others.popup import popup
import matplotlib.pyplot as plt
from others.Confusion_matrix import *
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Bidirectional, Dense, Conv1D, MaxPooling1D, Flatten, SimpleRNN, LSTM


def full_analysis():
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

            self.TCNN_Block_1 = TCNN_Block(in_channels=256, out_channels=512, kernel_size=3, init_dilation=2,
                                           num_layers=6)
            self.TCNN_Block_2 = TCNN_Block(in_channels=256, out_channels=512, kernel_size=3, init_dilation=2,
                                           num_layers=6)
            self.TCNN_Block_3 = TCNN_Block(in_channels=256, out_channels=512, kernel_size=3, init_dilation=2,
                                           num_layers=6)

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

    def extract():
        an = 0
        if an == 0:
            data_path, target_path = 'dataset/noise_train', 'dataset/clean_train'
            data_ = [wavfile.read(f'{data_path}/{noise}')[1] for noise in os.listdir(data_path)]
            target_ = [wavfile.read(f'{target_path}/{noise}')[1] for noise in os.listdir(target_path)]
            data_min_length = min([len(arr) for arr in data_])
            data = [final_noise[:data_min_length] for final_noise in data_]
            target_min_length = min([len(arr) for arr in target_])
            target = [final_noise[:target_min_length] for final_noise in target_][:len(data)]
            # np.save('pre_evaluated/data', data)
            # np.save('pre_evaluated/target', target)
        data, target = np.load('pre_evaluated/data.npy', allow_pickle=True), np.load('pre_evaluated/target.npy',
                                                                                     allow_pickle=True)
        return [data, target]

    def tsting(feat, target):
        global OUT

        def conv_tcnn():
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
                        DepthwiseSeparableConv(in_channels=out_channels, out_channels=in_channels,
                                               kernel_size=kernel_size,
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

                def forward(self, x):
                    return self.network(x)

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

            class TCNN(nn.Module):
                def __init__(self):
                    super(TCNN, self).__init__()
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

                    self.TCNN_Block_1 = TCNN_Block(in_channels=256, out_channels=512, kernel_size=3, init_dilation=2,
                                                   num_layers=6)
                    self.TCNN_Block_2 = TCNN_Block(in_channels=256, out_channels=512, kernel_size=3, init_dilation=2,
                                                   num_layers=6)
                    self.TCNN_Block_3 = TCNN_Block(in_channels=256, out_channels=512, kernel_size=3, init_dilation=2,
                                                   num_layers=6)

                    self.DConv2d_7 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 5), stride=(1, 2),
                                           padding=(1, 1),
                                           output_padding=(0, 0)),
                        nn.BatchNorm2d(num_features=64),
                        nn.PReLU()
                    )
                    self.DConv2d_6 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(3, 5), stride=(1, 2),
                                           padding=(1, 1),
                                           output_padding=(0, 0)),
                        nn.BatchNorm2d(num_features=32),
                        nn.PReLU()
                    )
                    self.DConv2d_5 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 5), stride=(1, 2),
                                           padding=(1, 1),
                                           output_padding=(0, 0)),
                        nn.BatchNorm2d(num_features=32),
                        nn.PReLU()
                    )
                    self.DConv2d_4 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(3, 5), stride=(1, 2),
                                           padding=(1, 1),
                                           output_padding=(0, 0)),
                        nn.BatchNorm2d(num_features=16),
                        nn.PReLU()
                    )
                    self.DConv2d_3 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 5), stride=(1, 2),
                                           padding=(1, 1),
                                           output_padding=(0, 1)),
                        nn.BatchNorm2d(num_features=16),
                        nn.PReLU()
                    )
                    self.DConv2d_2 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 5), stride=(1, 2),
                                           padding=(1, 2),
                                           output_padding=(0, 1)),
                        nn.BatchNorm2d(num_features=16),
                        nn.PReLU()
                    )
                    self.DConv2d_1 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(3, 5), stride=(1, 1),
                                           padding=(1, 2),
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

            conv = TCNN()
            conv

        def lstm():
            model = Sequential()
            model.add(LSTM(128, input_shape=lstm_X_train[1].shape))
            model.add(Dense(y_train.shape[1], activation='sigmoid'))
            model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
            model.fit(lstm_X_train, y_train, epochs=int(300), batch_size=10, verbose=1)
            y_predict = (model.predict(lstm_X_test)).flatten()
            return y_predict

        def cnn():
            print('comparison training started -- RNN model')
            model = Sequential()
            model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Conv1D(64, kernel_size=3, activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dense(y_train.shape[1], activation='linear'))
            model.compile(loss='mse', optimizer='adam')
            X_train_reshaped = np.expand_dims(X_train, axis=-1)
            X_test_reshaped = np.expand_dims(X_test, axis=-1)

            # Train the model
            model.fit(X_train_reshaped, y_train, epochs=300, batch_size=32, validation_data=(X_test_reshaped, y_test))
            y_predict = (model.predict(X_test_reshaped)).flatten()

            return y_predict

        def rnn():
            print('comparison training started -- RNN model')
            model = Sequential()
            model.add(SimpleRNN(64, input_shape=lstm_X_train[0].shape, activation='relu'))
            model.add(Dense(y_train.shape[1], activation='softmax'))
            model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
            model.fit(lstm_X_train, y_train, epochs=int(300), batch_size=32, verbose=0)
            y_predict = (model.predict(lstm_X_test)).flatten()
            return y_predict

        def bi_lstm():
            model = Sequential()
            model.add(Bidirectional(LSTM(128, input_shape=lstm_X_train[0].shape)))
            model.add(Dense(ln, activation='sigmoid'))
            model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
            model.fit(lstm_X_train, y_train, epochs=int(300), batch_size=10, verbose=1)
            y_predict = (model.predict(lstm_X_test)).flatten()
            return y_predict

        an = 0
        if an == 0:
            lp1, OUT = [0.4, 0.3, 0.2, 0.1], []
            print('testing_running...')
            ln = len(np.unique(target))
            for lp in lp1:
                met = []
                X_train, X_test, y_train, y_test = train_test_split(feat[:, :18775], target, test_size=lp,
                                                                    random_state=42)
                lstm_X_train = X_train.reshape(-1, X_train.shape[1], 1)
                lstm_X_test = X_test.reshape(-1, X_test.shape[1], 1)
                y_train, y_test = y_train.astype('int64'), y_test.astype('int64')
                model = PROP_TCNN()
                print(model)
                comp_ = [lstm, cnn, rnn, bi_lstm]
                for i in range(len(comp_)):
                    soln, ln = 10, len(np.unique(target))
                    # print('dataset-' + str(dt) + 'lp=' + str(lp) + str(comp_[i]))
                    y_t, pd = eval("{ i<=1 :(0.3,0.1),i==2:(0.5,0.2),i>2<=5 :(0.3,0.4),i==6 :(0.9,0.6)}[True]")
                    con = multi_confu_matrix(y_test, comp_[i](), y_t, pd, True)[0]
                    if i == len(comp_):
                        comp1 = []
                        for k in range(len(comp_)):
                            X_train, X_test, y_train, y_test = train_test_split(feat, target, test_size=lp,
                                                                                random_state=42)
                            lstm_X_train = X_train.reshape(-1, X_train.shape[1], 1)
                            lstm_X_test = X_test.reshape(-1, X_test.shape[1], 1)
                            comp1.append(multi_confu_matrix(y_test, lstm(), y_t, pd, True)[0])
                            # np.save('pre_evaluated/comp1', comp1)
                    met.append(con)
                OUT.append(met)
            OUT = array(OUT)
            # np.save('pre_evaluated/OUT', OUT)
        # OUT = np.load('pre_evaluated/OUT.npy')
        return OUT

    def ststs(a):
        b = np.empty([5])
        b[0] = np.min(a)
        b[1] = np.max(a)
        b[2] = np.mean(a)
        b[3] = np.median(a)
        b[4] = np.std(a)
        return b

    MS_SNSD = extract()
    coo2 = tsting(MS_SNSD[0], MS_SNSD[1])
    metrices = ['sensitivity', 'specificity', 'accuracy', 'precision', 'f_measure', 'mcc', 'npv', 'fpr', 'fnr']
    COO2 = np.load('pre_evaluated/OUTFF.npy', allow_pickle=True)
    stoi = [np.load('pre_evaluated/stoi__2db.npy', allow_pickle=True),
            np.load('pre_evaluated/stoi__5db.npy', allow_pickle=True)]
    pesq = [np.load('pre_evaluated/pesq_2db.npy', allow_pickle=True),
            np.load('pre_evaluated/pesq_5db.npy', allow_pickle=True)]
    ALG = ['lstm', 'cnn', 'rnn', 'bi_lstm', 'CONV_TCNN', 'PROP_TCNN']
    snr = ['-2db', '-5db']
    noise = ['AirConditioner', 'Babble', 'Munching', 'Average']
    clr, lp = ['b', 'y', 'g', 'm', 'c', 'r', '#a6e32b', '#f216ef', '#820c0c'], ['60', '70', '80', '90']
    print('MS-SNSD --Results')
    for i in range(len(metrices)):
        plt.figure()
        value1 = []
        for k in range(len(ALG)):
            value = []
            for j in range(4):
                value.append(COO2[j, k, i])
            value1.append(value)
        if i == 2:
            print('statistical analysis')
            print((pd.DataFrame(np.array([ststs(x) for x in value1]),
                                columns=['WORST', 'BEST', 'MEAN', 'MEDN', 'STND DEV'], index=ALG)).to_markdown())
        br1 = np.arange(4)
        W = 0.10
        for pt in range(len(value1)):
            br1 = np.arange(4) if pt == 0 else [x + W for x in br1]
            plt.bar(br1, value1[pt], color=clr[pt], width=W,
                    edgecolor='grey', label=ALG[pt])

        plt.subplots_adjust(bottom=0.2)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3)
        plt.xlabel('Learning Percent (%)', fontweight='bold')
        plt.ylabel(metrices[i], fontweight='bold')
        plt.xticks([r + 0.13 for r in range(4)],
                   ['60', '70', '80', '90'])

    Text = [[print(f'learning_rate{lp[i]}'), print(pd.DataFrame(COO2[i], columns=metrices, index=ALG).to_markdown())]
            for i
            in range(len(COO2))]
    stoii = [[print(f'STOI{snr[i]}'), print(pd.DataFrame(stoi[i], columns=noise, index=ALG).to_markdown())] for i
             in range(len(stoi))]
    pesqq = [[print(f'PESQ{snr[i]}'), print(pd.DataFrame(pesq[i], columns=noise, index=ALG).to_markdown())] for i
             in range(len(stoi))]
    plt.show(block=True)


popup(full_analysis, results.result)
