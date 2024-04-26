
from tempurranet.model.old.impl import UNetConvLSTM_34
import torch
from config import tempurra_tusimple

from torch.utils.data import DataLoader
from tempurranet.datasets.TUSimple import TuSimple

root = "../../tempurranet/data/TUSimple"
split = "train_val"
cfg = tempurra_tusimple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetConvLSTM_34.TempURRA(n_classes=2, cfg=tempurra_tusimple, lstm_layers=5, hidden_dim=[512, 512, 512, 512, 512], kernel_size=(3, 3)).to(device)

dataset = TuSimple(root, split=split, processes=tempurra_tusimple.dataset['train']['processes'], cfg=tempurra_tusimple)
tu_train_dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

for batch_idx, batch in enumerate(tu_train_dataloader):
    ...
    # labels = batch['lanes'].view(32, -1)
    # data = [x.permute((0, 3, 1, 2)) for x in batch['prev_frames']]
    # data.append(batch['img'])
    # data = [t.to(device) for t in data]
    #
    # output = model(data)

    # print("br")
    # summary(model, data)
    # if not input_size or any(size <= 0 for size in flatten(input_size)):
    # input()
