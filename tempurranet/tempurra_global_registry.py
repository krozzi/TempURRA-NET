from tempurranet.util.registry import Registry, build_from_cfg
import torch.nn as nn

import torch
from functools import partial
import numpy as np
import random
from mmcv.parallel import collate

# borrowed and altered from https://github.com/Turoad/CLRNet/tree/main

# model registry
BACKBONES = Registry('backbones')
AGGREGATORS = Registry('aggregators')
HEADS = Registry('heads')
NECKS = Registry('necks')
NETS = Registry('nets')

# engine util registry
TRAINER = Registry('trainer')
EVALUATOR = Registry('evaluator')

# dataset stuff
DATASETS = Registry('datasets')
PROCESS = Registry('process')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbones(cfg):
    return build(cfg.backbone, BACKBONES, default_args=dict(cfg=cfg))


def build_necks(cfg):
    return build(cfg.necks, NECKS, default_args=dict(cfg=cfg))


def build_aggregator(cfg):
    return build(cfg.aggregator, AGGREGATORS, default_args=dict(cfg=cfg))


def build_heads(cfg):
    return build(cfg.heads, HEADS, default_args=dict(cfg=cfg))


def build_head(split_cfg, cfg):
    return build(split_cfg, HEADS, default_args=dict(cfg=cfg))


def build_trainer(cfg):
    return build(cfg.trainer, TRAINER, default_args=dict(cfg=cfg))


def build_evaluator(cfg):
    return build(cfg.evaluator, EVALUATOR, default_args=dict(cfg=cfg))


def build_net(cfg):
    return build(cfg.net, NETS, default_args=dict(cfg=cfg))


def build_dataset(split_cfg, cfg):
    return build(split_cfg, DATASETS, default_args=dict(cfg=cfg))


def worker_init_fn(worker_id, seed):
    worker_seed = worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(split_cfg, cfg, is_train=True):
    if is_train:
        shuffle = True
    else:
        shuffle = False

    dataset = build_dataset(split_cfg, cfg)

    init_fn = partial(worker_init_fn, seed=cfg.seed)

    samples_per_gpu = cfg.batch_size // cfg.gpus

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        worker_init_fn=init_fn)

    return data_loader
