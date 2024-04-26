from typing import List

import torch
import config
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import operator
from ..unet import *
from ..lstm import *
from tempurranet.tempurra_global_registry import build_backbones


class UNet_ConvLSTM(nn.Module):

    def __init__(self, n_classes: int, hidden_dims: List[int], cfg, output_points: int = 358, layers: int = 5,
                 batch_first: bool = True, ):
        """
        Init for TempURRANET
        :param n_classes: Number of output classes for the U-NET (seg. mask)
        :param hidden_dims: list of hidden layers dimensions used to define the convlstm architecture (e.g [512, 512]).
        :param cfg: Config file for the net
        :param output_points: Number of points to output using FCL
        :param layers: Number of LSTM layers
        :param batch_first: Whether the input tensors have batch first i.e. (b, t, c, h, w) or (t, b, c, h, w)
        """

        super(UNet_ConvLSTM, self).__init__()
        self.up1 = up(512 * 2, 256)
        self.up2 = up(256 * 2, 128)
        self.up3 = up(128 * 2, 64)
        self.up4 = up(64 * 2, 32)
        self.outc = outconv(32, n_classes)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.convlstm = ConvLSTM(input_size=(4, 8),
                                 input_dim=512,
                                 hidden_dim=hidden_dims,
                                 kernel_size=(3, 3),
                                 num_layers=layers,
                                 batch_first=batch_first,
                                 bias=True)

        self.fc1 = nn.Linear(in_features=8192, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=output_points)

        self.inReLU = nn.ReLU(inplace=True)
        self.ReLU = nn.ReLU()

        self.bn = nn.BatchNorm1d(2048)

        self.backbone = build_backbones(cfg)

    def forward(self, x: List):
        """
        Forward pass of the model. We expect the input to be a list of batched samples, each element being one
        step/frame. All batch sizes are expected to be the same.
        :param x: List of batched samples.
        :return: Output of the model, a series of a bunch of points.
        """

        # contains a list of features maps for each block of the encoder (e.g. dim of [512, 128, ...])
        cat_encodings = []
        lstm_encodings = None
        for idx, batched_sample in enumerate(x):
            features = self.backbone(batched_sample)
            if idx == len(x) - 1:
                for feature in features:
                    cat_encodings.append(feature)
            if lstm_encodings is None:
                lstm_encodings = features[-1].unsqueeze(1)
            else:
                lstm_encodings = torch.cat((lstm_encodings, features[-1].unsqueeze(1)), dim=1)

        out, _ = self.convlstm(lstm_encodings)
        out = out[0][:, -1, :, :, :]

        out = self.up1(out, cat_encodings[3])
        out = self.up2(out, cat_encodings[2])
        out = self.up3(out, cat_encodings[1])
        out = self.up4(out, cat_encodings[0])
        out = self.upsample(out)
        out = self.outc(out)

        # out = torch.softmax(out, dim=0)
        # out = torch.argmax(out, dim=1)  # (10, 128, 256)


        # conv to point conversion
        # out = out.view(out.shape[0], -1)
        #
        # out = out.to(torch.float32)
        # out = self.inReLU(self.fc1(out))
        # out = self.bn(out)
        # out = self.inReLU(self.fc2(out))
        # out = self.inReLU(self.fc3(out))

        return out
