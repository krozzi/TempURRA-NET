import cv2
import os
import json

from tempurranet.tempurra_global_registry import DATASETS
from tempurranet.datasets.preprocessing.preprocesser import Process
from config.tempurra_tusimple import logger
import random
from typing import List, Dict
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

SPLIT_FILES = {
    'train': 'fixed/ord_train.json',
    'val': 'fixed/ord_val.json',
    'train_val': 'fixed/ord_train_val.json',
    'test': 'fixed/ord_test.json'
}


def interpolate_points(points, N):
    interp = points.copy()
    i = 0
    pt_len = len(interp)
    print(pt_len, i, interp)

    while N > 0:

        if i >= pt_len:
            pt_len = len(interp)
            i = 0
            continue

        nxt_pt = interp[i + 1]
        cur_pt = interp[i]

        interp.insert(i + 1, [
            cur_pt[0] + 0.5 * (nxt_pt[0] - cur_pt[0]),
            cur_pt[1] + 0.5 * (nxt_pt[1] - cur_pt[1]),
        ])

        i += 2
        N -= 1

    return interp

@DATASETS.register_module
class TuSimple(Dataset):
    def __init__(self, data_root, split, processes=None, cfg=None):
        self.cfg = cfg
        self.logger = logger
        self.data_root = data_root
        self.training = 'train' in split
        if processes is not None:
            self.processes = Process(processes, cfg)
        self.data_infos: List[Dict] = []

        self.anno_files = SPLIT_FILES[split]
        self.load_annotations()
        self.h_samples = list(range(160, 720, 10))

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        img = cv2.imread(data_info['img_path'])

        prev_frames = data_info['prev_frames']
        prev_images = []

        for frame in prev_frames:
            prev_images.append(cv2.imread(frame))

        sample = data_info.copy()
        sample.update({'img': img})
        sample.update({'prev_frames': prev_images})

        try:
            sample = self.processes(sample)
        except AttributeError:
            pass
        # meta = {'full_img_path': data_info['img_path']}
        # meta = DC(meta, cpu_only=True)
        # sample.update({'meta': meta})

        return sample

    def load_annotations(self):
        self.logger.info("Loading TUSimple annotations...")
        folder = 'train' if self.training else 'test'

        split_root = os.path.join(self.data_root, folder)
        fixed_json_anno_pth = os.path.join(split_root, self.anno_files)

        with open(fixed_json_anno_pth, 'r') as lane_json:
            dataset = lane_json.read().splitlines()
            dataset = [json.loads(x) for x in dataset]

            for data in dataset:
                h_samples = data['h_samples']
                lane_data = []

                for lane in data['lanes']:
                    for coord in lane:
                        lane_data.append(coord)

                    #
                    #
                    # if len(lane) == len(h_samples):
                    #     for x, y in zip(lane, h_samples):
                    #         if x == -2 or y == -2:
                    #             continue
                    #         lane_data.append((x, y))


                # interpolate stuff used to ber here

                self.data_infos.append({
                    'img_path': os.path.join(split_root, data['raw_file']),
                    'prev_frames': tuple(map(lambda x: os.path.join(split_root, x), data['prev_frames'])),
                    'lanes': lane_data
                })

        if self.training:
            random.shuffle(self.data_infos)
