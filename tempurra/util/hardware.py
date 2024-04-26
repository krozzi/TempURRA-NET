import torch
from config.tempurra_tusimple import logger

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if device == torch.device("cuda"):
    logger.info("Using device:", torch.cuda.get_device_name(0))
else:
    logger.info("Using device:", device)


def set_gpu_number(n_gpu):
    global device
    device = torch.device("cuda:{}".format(n_gpu)) if torch.cuda.is_available() else torch.device("cpu")
