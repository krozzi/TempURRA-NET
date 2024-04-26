import time

from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch

from tempurranet.datasets.TUSimple import TuSimple
from tempurranet.model.new.impl import ISEFModel2
from tempurranet.model.new.impl.ISEFModel2 import UNet_ConvLSTM
from tempurranet.util.train_util import AverageMeter, ProgressMeter, save_model_checkpoint
from tempurranet.util.hardware import device
from tempurranet.model.new.losses import NetLoss
from tempurranet.util import visualize as vs
from tempurranet.util import utjson

from config import tempurra_tusimple_eval
import json
import os


global_batch = 0


def validate(val_loader, model, criterion, log_every=1):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc', ':6.4f')
    f1 = AverageMeter('F1', ':6.4f')
    prec = AverageMeter('Prec', ':6.4f')
    rec = AverageMeter('Recall', ':6.4f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, acc, f1, prec, rec],
        prefix='Test: ')

    # model.eval() evaluate mode highly decreases performance
    model.train()

    correct = 0
    error = 0
    precision = 0.
    recall = 0.
    with torch.no_grad():
        end = time.time()
        for batch_no, batch in enumerate(val_loader):
            batch_time.update(time.time() - end)

            data = [x.permute((0, 3, 1, 2)) for x in batch['prev_frames']]
            data.append(batch['img'])

            data = [t.to(device) for t in data]

            labels = batch['lanes']
            labels = labels.type(torch.LongTensor).to(device)

            # compute output
            output = model(data)
            # compute loss
            loss = criterion(output, labels)
            losses.update(loss.item(), labels.size(0))
            # compute f1 score
            f, (p, r) = f1_score(output, labels)
            f1.update(f)
            prec.update(p)
            rec.update(r)
            # compute accuracy
            acc.update(pixel_accuracy(output, labels), labels.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_no % log_every == 0:
                progress.display(batch_no)

        return acc.avg

def pixel_accuracy(prediction: torch.Tensor, target: torch.Tensor):
    """
        Computes simple pixel-wise accuracy measure
        between target lane and prediction map; this
        measure has little meaning (if not backed up
        by other metrics) in tasks like this where
        there's a huge unbalance between 2 classes
        (background and lanes pixels).
    """
    # get prediction positive channel (lanes)
    out = (prediction[:, 1, :, :] > 0.).float()
    return (out == target).float().mean().item()


def f1_score(output, target, epsilon=1e-7):
    # output has to be passed though sigmoid and then thresholded
    # this way we directly threshold it efficiently
    output = torch.sigmoid(output)
    probas = (output[:, 1, :, :] > 0.).float()

    TP = (probas * target).sum(dim=1)
    precision = TP / (probas.sum(dim=1) + epsilon)
    recall = TP / (target.sum(dim=1) + epsilon)
    f1 = (2 * precision * recall) / (precision + recall + epsilon)
    f1 = f1.clamp(min=epsilon, max=1-epsilon)
    return f1.mean().item(), (precision.mean().item(), recall.mean().item())


def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = init_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter()

    if os.listdir("../train/data") == (None or []):
        print("No data found, creating new JSON.")
        utjson.create_json("../train/data/metrics.json")
    else:
        print("Existing data found, continuing with setup.")

    root = "../../../../tempurranet/data/TUSimple"
    split = "test"
    cfg = tempurra_tusimple_eval
    init_lr = 1e-5

    model = ISEFModel2.UNet_ConvLSTM(cfg=cfg, n_classes=2, hidden_dims=[512, 512, 512], layers=3).to(
        device=device)

    model.load_state_dict(torch.load("../train/model.torch", map_location=device)['model_state_dict'])

    dataset = TuSimple(root, split=split, processes=tempurra_tusimple_eval.dataset['val']['processes'], cfg=tempurra_tusimple_eval)
    tu_train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    model = model.to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), init_lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg.total_iter))
    criterion = NetLoss(0.1, device).to(device)
    # criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.02, 1.02])).to(device)

    losses_ = []
    accs = []
    f1s = []
    test_losses = []

    accs = validate(tu_train_dataloader, model, criterion, log_every=1)
    print("done")

