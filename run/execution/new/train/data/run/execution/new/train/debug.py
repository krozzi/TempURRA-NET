from tempurranet.model.old.impl import UNetConvLSTM_34
import torch
from tempurranet.model.new.impl import ISEFModel2
from config import tempurra_tusimple
import numpy as np
import cv2
from torch.utils.data import DataLoader
from tempurranet.datasets.TUSimple import TuSimple
import tempurranet.util.visualize as vs
from tempurranet.util import train_util

root = "../../../../tempurranet/data/TUSimple"
split = "train_val"
cfg = tempurra_tusimple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ISEFModel2.UNet_ConvLSTM(cfg=tempurra_tusimple, n_classes=2, hidden_dims=[512, 512, 512], layers=3).to(device=device)
model.load_state_dict(torch.load("model.torch", map_location=device)['model_state_dict'])
dataset = TuSimple(root, split=split, processes=tempurra_tusimple.dataset['train']['processes'], cfg=tempurra_tusimple)
tu_train_dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

model.to(device)

# vs.display_raw(out[:, 1, :, :].unsqueeze(1)[0].unsqueeze(1))



# for rowid, row in enumerate(final_out_indiv_test):
#
#     if rowid % 10 == 0:
#         print("yo")
#
#         start = -1
#         end = -1
#         middle = -1
#
#         for colid, col in enumerate(row):
#             if col > 0 and start == -1:
#                 print('start detected')
#                 start = int(colid)
#             if (not (col > 0)) and start != -1:
#                 print("end detected")
#                 end = int(colid)
#                 middle = int((start + end ) // 2)
#                 coords.append([middle, int(rowid)])
#                 start = -1
#                 end = -1
#                 middle = -1

# for idx, frame in enumerate(out[:, 1, :, :].unsqueeze(1)[0].unsqueeze(1)):
#     frame = frame.detach().cpu().numpy().transpose(1, 2, 0).copy()
#     for coord in coords:
#         cv2.circle(frame, (coord[0], coord[1]), 1, (255, 255, 25) , 2)
#         print("coord drawn")
#     cv2.imshow(f"Image ID#{idx}", frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

for batch_idx, batch in enumerate(tu_train_dataloader):
    labels = batch['lanes'].view(tempurra_tusimple.batch_size, -1)
    data = [x.permute((0, 3, 1, 2)) for x in batch['prev_frames']]
    data.append(batch['img'])
    data = [t.to(device) for t in data]
    out = model(data)
    print("asdf")