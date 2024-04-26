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

from config import tempurra_tusimple
import json
import os

global_batch = 0

def train(train_loader: DataLoader, model: UNet_ConvLSTM, criterion, optimizer, scheduler, epoch, writer, log_every=1):

    """
    Do a training step, iterating over all batched samples
    as returned by the DataLoader passed as argument.
    Various measurements are taken and returned, such as
    accuracy, loss, precision, recall, f1 and batch time.
    """

    global global_batch
    batch_time = AverageMeter('BatchTime', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc', ':6.4f')
    f1 = AverageMeter('F1', ':6.4f')
    prec = AverageMeter('Prec.', ':6.4f')
    rec = AverageMeter('Recall', ':6.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc, f1, prec, rec],
        prefix="Epoch: [{}]".format(epoch))

    model.train().cuda()

    start = time.time()

    for batch_no, batch in enumerate(train_loader):
        data_time.update(time.time() - start)

        data = [x.permute((0, 3, 1, 2)) for x in batch['prev_frames']]
        data.append(batch['img'])

        data = [t.to(device) for t in data]

        labels = batch['lanes']
        labels = labels.type(torch.LongTensor).to(device)

        output = model(data)

        loss = criterion(output, labels)
        # loss = Variable(loss, requires_grad=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.detach()
        # record loss, dividing by sample size
        losses.update(loss.item(), labels.size(0))

        # labels = labels.float()
        accuracy = pixel_accuracy(output, labels)
        acc.update(accuracy, labels.size(0))

        batch_time.update(time.time() - start)
        start = time.time()

        f, (p, r) = f1_score(output, labels)

        f1.update(f)
        prec.update(p)
        rec.update(r)

        writer.add_scalar("Batch-wise Loss", float(loss.item()), global_batch)
        writer.add_scalar("Batch-wise Accuracy", float(accuracy), global_batch)
        writer.add_scalar("Batch-wise F1 Score", float(f), global_batch)
        writer.add_images("Batch-wise Argmax Prediction",
                          torch.argmax(torch.softmax(output, dim=0), dim=1).unsqueeze(1), global_batch)
        writer.add_images("Batch-wise Softmax Prediction", torch.softmax(output, dim=0)[:, 1, :, :].unsqueeze(1), global_batch)
        writer.add_images("Batch-wise Raw Prediction", output[:, 1, :, :].unsqueeze(1), global_batch)

        if batch_no % log_every == 0:
            print("----------------------------------------------------")
            print("Output min", output.min().item(), "Output (softmax-ed) sum:", (output > 0.).float().sum().item(),
                  "Output max:", torch.max(output).item())
            print("Targets sum:", labels.sum())  # , "Targets max:", torch.max(batched_targets))
            print("Base acc:{} - base prec: {} - base recall: {} - base f1: {}"
                  .format(pixel_accuracy(output, labels), p, r, f))
            progress.display(batch_no)
            print("----------------------------------------------------")

        global_batch += 1
    scheduler.step()

    return losses.avg, acc.avg, f1.avg


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

    if os.listdir("./data") == (None or []):
        print("No data found, creating new JSON.")
        utjson.create_json("./data/metrics.json")
    else:
        print("Existing data found, continuing with setup.")

    root = "../../../../tempurranet/data/TUSimple"
    split = "train_val"
    cfg = tempurra_tusimple
    init_lr = 1e-5

    model = ISEFModel2.UNet_ConvLSTM(cfg=cfg, n_classes=2, hidden_dims=[512, 512, 512], layers=3).to(
        device=device)

    model.load_state_dict(torch.load("model.torch", map_location=device)['model_state_dict'])

    dataset = TuSimple(root, split=split, processes=tempurra_tusimple.dataset['train']['processes'],
                       cfg=tempurra_tusimple)
    tu_train_dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    model = model.to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), init_lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg.total_iter))
    criterion = NetLoss(0.1, device).to(device)
    # criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.02, 1.02])).to(device)

    losses_ = []
    accs = []
    f1s = []
    test_losses = []


    for epoch in range(cfg.epochs):

        loss_val, a, f = train(tu_train_dataloader, model, criterion, optimizer, scheduler, epoch, writer, log_every=19)

        losses_.append(loss_val)
        accs.append(a)
        f1s.append(f)

        utjson.append_json_exit({
            "epoch": epoch,
            "loss": loss_val,
            "accuracy": a,
            "f1score": f1s
        }, "step", "data/metrics.json")

        writer.add_scalar("Loss", loss_val, epoch)
        writer.add_scalar("Accuracy", a, epoch)
        writer.add_scalar("F1 Score", f, epoch)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
            # 'tr_loss': tr_loss,
            # 'ev_loss': ev_loss
        }, "model.torch")

        # torch.save({'model_state_dict': model.state_dict()}, "test.torch")

        # save_model_checkpoint(model, 'unet_convlstm.torch', epoch=epoch)

    print("Saving loss values to json..")
    with open('losses.json', 'w') as f, open('acc.json', 'w') as ff, open('f1score.json', 'w') as fff:
        json.dump(losses_, f)
        json.dump(accs, ff)
        json.dump(f1s, fff)