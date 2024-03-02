import torch
import torch.nn as nn
from tempurranet.model.unet import *
from tempurranet.model.lstm import *
from tempurranet.tempurra_global_registry import build_backbones


def reshape_tensor(input_tensor, target_shape):
    """
    Reshapes the input tensor to the target shape without adding junk data.

    Parameters:
        input_tensor (torch.Tensor): Input tensor to be reshaped.
        target_shape (tuple): Target shape for the tensor.

    Returns:
        torch.Tensor: Reshaped tensor.
    """
    # Calculate the number of times to repeat the tensor completely
    num_repeats = target_shape[0] // input_tensor.shape[0]

    # Calculate the remaining portion
    remainder = target_shape[0] % input_tensor.shape[0]

    # Repeat the tensor completely
    new_tensor = torch.cat([input_tensor] * num_repeats, dim=0)

    # Append the remaining portion
    if remainder > 0:
        new_tensor = torch.cat([new_tensor, input_tensor[:remainder]], dim=0)

    return new_tensor

class TempURRA(nn.Module):
    def __init__(self, n_classes, cfg, lstm_layers: int = 2, hidden_dim: list = [512, 512],
                 kernel_size: tuple = (3, 3), n_output: int = 358):
        super(TempURRA, self).__init__()
        self.cfg = cfg

        # modules
        self.backbone = build_backbones(cfg)
        self.lstm = ConvLSTM(input_size=(4, 8),
                             input_dim=512,
                             hidden_dim=hidden_dim,
                             kernel_size=kernel_size,
                             num_layers=lstm_layers,
                             batch_first=True,
                             bias=True,
                             return_all_layers=False)

        # unet stuff
        self.n_classes = n_classes
        self.bilinear = cfg.net['bilinear']

        factor = 2 if self.bilinear else 1

        self.up1 = (Up(1024, 512 // factor, self.bilinear))
        self.up2 = (Up(512, 256 // factor, self.bilinear))
        self.up3 = (Up(256, 128 // factor, self.bilinear))
        self.up4 = (Up(128, 64, self.bilinear))
        self.outc = (OutConv(64, n_classes))
        self.linear = nn.Linear(4096, n_output)

    def forward(self, x):
        encoded_images = []

        for i, batched_sample in enumerate(x):
            features = self.backbone(batched_sample)
            encoded_images.append(features)

        # includes the stacks of each dim; stacked 64, 128, 256, 512, etc. which is for each frame
        batched_encoded_sequences = []

        for i in range(len(encoded_images[0])):
            batched_sequences = torch.stack([image[i] for image in encoded_images], dim=1)
            batched_encoded_sequences.append(batched_sequences)

        decoder_batched_sequences = list(reversed(batched_encoded_sequences))

        # runs LSTM on the last stack, most dims (512), smallest res.
        output, _ = self.lstm(batched_encoded_sequences[-1])
        output = output[0][:, -1, :, :, :]

        x = self.up1(output, reshape_tensor(decoder_batched_sequences[0][-1], (32,)))
        x = self.up2(x, reshape_tensor(decoder_batched_sequences[1][-1], (32,)))
        x = self.up3(x, reshape_tensor(decoder_batched_sequences[2][-1], (32,)))
        x = self.up4(x, reshape_tensor(decoder_batched_sequences[3][-1], (32,)))
        x = self.outc(x)
        # encoder

        # convert to points
        x = x.reshape(-1)
        x = self.linear(x)


        # (b, t, c, h, w), should be concated along channel dim
        return x
