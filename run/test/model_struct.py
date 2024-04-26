from tempurranet.model.old.impl import UNetConvLSTM_34
import torch
from tempurranet.model.new.impl import ISEFModel2
from config import tempurra_tusimple
import numpy as np
import cv2
from torch.utils.data import DataLoader
from tempurranet.datasets.TUSimple import TuSimple
import tempurranet.util.visualize as vs

root = "../../tempurranet/data/TUSimple"
split = "train_val"
cfg = tempurra_tusimple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = UNetConvLSTM_34.TempURRA(n_classes=2, cfg=tempurra_tusimple, lstm_layers=5, hidden_dim=[512, 512, 512, 512, 512], kernel_size=(3, 3)).to(device)
model = ISEFModel2.UNet_ConvLSTM(cfg=tempurra_tusimple, n_classes=2, hidden_dims=[512, 512, 512], layers=3).to(device=device)
dataset = TuSimple(root, split=split, processes=tempurra_tusimple.dataset['train']['processes'], cfg=tempurra_tusimple)
tu_train_dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

for batch_idx, batch in enumerate(tu_train_dataloader):
    labels = batch['lanes'].view(tempurra_tusimple.batch_size, -1)
    data = [x.permute((0, 3, 1, 2)) for x in batch['prev_frames']]
    data.append(batch['img'])
    data = [t.to(device) for t in data]
    out = model(data)
    ...