import time

from tempurranet.model.old.impl import UNetConvLSTM_34
import torch
from config import tempurra_tusimple

from torch.utils.data import DataLoader
from tempurranet.datasets.TUSimple import TuSimple
from torch.optim import lr_scheduler

def train(epoch, model, train_loader, device, optimizer, criterion):
    since = time.time()
    model.train()
    for batch_idx, batch in enumerate(train_loader):

        target = batch['lanes'].view(32, -1).to(device)
        data_list = batch['img'].detach().clone()
        data = torch.Tensor(len(data_list), 6, 3, 128, 256).to(device)
        for i in range(len(batch['img'])):
            tmp = []
            for j in range(5):
                tmp.append(batch['prev_frames'][j][i].permute(2, 0, 1))
            data[i] = torch.cat((data_list[i].unsqueeze(0), torch.stack(tmp)))

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    time_elapsed = time.time() - since
    print('Train Epoch: {} complete in {:.0f}m {:.0f}s'.format(epoch,
        time_elapsed // 60, time_elapsed % 60))
    torch.save(model.state_dict(), 'UNETMODEL.pth')


if __name__ == '__main__':
    torch.manual_seed(1111)

    root = "../../../tempurranet/data/TUSimple"
    split = "train_val"
    cfg = tempurra_tusimple

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetConvLSTM_34.TempURRA(n_classes=2, cfg=tempurra_tusimple, lstm_layers=5,
                                     hidden_dim=[512, 512, 512, 512, 512], kernel_size=(3, 3)).to(device)

    dataset = TuSimple(root, split=split, processes=tempurra_tusimple.dataset['train']['processes'],
                       cfg=tempurra_tusimple)
    tu_train_dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    class_weight = torch.Tensor([0.02, 1.02])
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight).to(device)
    best_acc = 0

    # pretrained_dict = torch.load(config.pretrained_path)
    # model_dict = model.state_dict()
    #
    # pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
    # model_dict.update(pretrained_dict_1)
    # model.load_state_dict(model_dict)

    for epoch in range(1, tempurra_tusimple.epochs + 1):
        scheduler.step()
        train(epoch, model, tu_train_dataloader, device, optimizer, criterion)