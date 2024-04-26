import torch.nn as nn
import torch
import torch.nn.functional as F

# huge thanks to ultra fast lane det.


class ParsingRelationLoss(nn.Module):
    def __init__(self):
        super(ParsingRelationLoss, self).__init__()

    def forward(self, logits):
        n, c, h, w = logits.shape
        loss_all = []
        for i in range(0, h-1):
            loss_all.append(logits[:, :, i, :] - logits[:, :, i+1, :])

        loss = torch.cat(loss_all)
        return torch.nn.functional.smooth_l1_loss(loss, torch.zeros_like(loss))


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):

        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss


# class ClassificationLoss(nn.Module):
#     def __init__(self):
#         super(ClassificationLoss, self).__init__()
#
#     def forward(self, logits, labels):
#


class NetLoss(nn.Module):
    def __init__(self, gamma, device, ignore_lb=255, *args, **kwargs):
        self.gamma = gamma
        self.device = device
        self.ignore_lb = ignore_lb
        super(NetLoss, self).__init__()

    def forward(self, logits, labels):
        # logits = logits.float().to(self.device)
        # labels = labels.float().to(self.device)
        return (
                SoftmaxFocalLoss(self.gamma)(logits, labels.type(torch.LongTensor).to(self.device)).to(self.device) +
                ParsingRelationLoss()(logits).to(self.device) +
                1.5 * nn.CrossEntropyLoss(weight=torch.FloatTensor([0.02, 1.02])).cuda()(logits.to(self.device), labels.to(self.device))
        )
