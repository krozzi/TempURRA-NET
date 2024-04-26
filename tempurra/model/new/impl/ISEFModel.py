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
        self.up2 = up(256* 2, 128)
        self.up3 = up(128, 64)
        self.up4 = up(64, 32)
        self.outc = outconv(32, n_classes)

        self.convlstm = ConvLSTM(input_size=(4, 8),
                                 input_dim=512,
                                 hidden_dim=hidden_dims,
                                 kernel_size=(3, 3),
                                 num_layers=layers,
                                 batch_first=batch_first,
                                 bias=True)

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
        for i, batched_sample in enumerate(x):
            features = self.backbone(batched_sample)
            if len(cat_encodings) == 0:
                for feature in features:
                    cat_encodings.append(feature)
                lstm_encodings = features[-1].unsqueeze(1)
            else:
                for idx, feature in enumerate(features):
                    cat_encodings[idx] = torch.cat((cat_encodings[idx], feature), dim=0)
                lstm_encodings = torch.cat((lstm_encodings, features[-1].unsqueeze(1)), dim=1)

        output, _ = self.convlstm(lstm_encodings)
        output = output[0][:, -1, :, :, :]

        output = self.up1(output, cat_encodings[3])
        output = self.up2(output, cat_encodings[2])
        output = self.up3(output, cat_encodings[1])
        output = self.up4(output, cat_encodings[0])
        output = self.outc(output)

        avged_output = torch.mean(output, dim=0)

        return output