import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.lines import LineString
from numpy import ndarray, dtype
import torch
from torch import Tensor
import math

from tempurranet.tempurra_global_registry import PROCESS
from typing import List, Any, Dict


def lane_to_linestring(lanes: List[int]) -> List[LineString]:
    """
    Converts a series of points into a series of lane strings.
    :param lanes: A series of points describing a lane line.
    :return: A series of line strings representing the lane lines.
    """
    lines = []
    for idx, lane in enumerate(lanes):
        if idx + 1 < len(lanes):
            lines.append(LineString((lane, lanes[idx+1])))

    return lines


def linestring_to_lanes(linestring: List[LineString], cfg) -> ndarray[Any, dtype[Any]]:
    """
    Convert linestring into lane points.
    :param linestring: A list of lane strings representing a lane.
    :return: A series of points describing a lane.
    """
    lane_points = []
    for line in linestring:
        lane_points.append([line.coords[0][0] / cfg.img_w, line.coords[0][1] / cfg.img_h])
        # lane_points.append([line.coords[0][0], line.coords[0][1]])
    return np.array(lane_points)


def lane_to_heatmap(lane:ndarray, cfg) -> Tensor:
    empty = torch.zeros((cfg.img_h, cfg.img_w))
    for i in lane:
        empty[max(0, min(int(i[1] * cfg.img_h), cfg.img_h - 1))][max(0, min(math.floor(i[0] * cfg.img_w), cfg.img_w - 1))] = 1
    return empty


@PROCESS.register_module
class GenerateLaneLine(object):
    def __init__(self, transforms: List[Dict] = None, training: bool = True, cfg=None) -> None:
        """
        Image and lane line transformer for dataset augmentation.
        :param transforms: A list of desired transformations.
        :param cfg: Config object.
        :param training: Training mode or no.
        """
        self.transforms = transforms
        self.training = training
        self.cfg = cfg

        if transforms is not None:
            img_transforms = []
            for aug in transforms:
                p = aug['p']
                if aug['name'] != 'OneOf':
                    img_transforms.append(
                        iaa.Sometimes(p=p, then_list=getattr(iaa, aug['name'])(**aug['parameters']))
                    )
                else:
                    img_transforms.append(
                        iaa.Sometimes(
                            p=p,
                            then_list=iaa.OneOf([
                                getattr(iaa, aug_['name'])(**aug_['parameters']) for aug_ in aug['transforms']
                            ])))
        else:
            img_transforms = []
        self.transform = iaa.Sequential(img_transforms)

    def __call__(self, sample: dict) -> dict:
        """
        Augments data.
        :param sample: A datapoint containing the image and lane points.
        :return: An altered input with transformations applied to the image and lane point.
        """

        if self.training:
            sample_copy = sample.copy()

            img, lns = self.transform(image=sample_copy['img'].copy().astype(np.uint8),
                                      line_strings=lane_to_linestring(sample_copy['lanes'].copy()))

            prev_frames = [self.transform(image=prev_frame.copy().astype(np.uint8)) for prev_frame in sample_copy['prev_frames']]
            sample_copy.update({'prev_frames': [img.astype(np.float32) / 255. for img in prev_frames]})

            sample_copy.update({'img': img.astype(np.float32) / 255.})
            sample_copy.update({'lanes': lane_to_heatmap(linestring_to_lanes(lns, self.cfg), self.cfg)})
            # sample_copy.update({'lanes': linestring_to_lanes(lns, self.cfg)})
            return sample_copy

        return sample
