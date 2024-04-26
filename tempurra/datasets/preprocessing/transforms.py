from typing import Sequence, Union
import numpy as np
import torch
from tempurranet.tempurra_global_registry import PROCESS


def to_tensor(data: Union[torch.Tensor, np.ndarray, Sequence, int, float]) -> torch.Tensor:
    """
    Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    :param data: Data to be converted.
    :return: Tensor-converted data
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@PROCESS.register_module
class ToTensor(object):

    def __init__(self, keys: Sequence[str] = ['img', 'mask'], cfg=None):
        """
        Convert some results to :obj:`torch.Tensor` by given keys.
        :param keys: Keys for values that need to be converted to Tensor.
        :param cfg: Config object
        """
        self.keys = keys

    def __call__(self, sample):
        """
        Applies transformations to the input sample and returns the modified data.

        :param sample: A dictionary containing the input sample data.
        :return: A dictionary with the modified data after applying transformations.
        """
        data = {}  # Initialize an empty dictionary to store the modified data

        # Check if the input image has less than 3 dimensions (channels, height, width)
        if len(sample['img'].shape) < 3:
            sample['img'] = np.expand_dims(sample['img'], -1)  # Expand dimensions to add a channel dimension

        # Iterate over the keys in the sample dictionary
        for key in self.keys:
            # Check if the key is one of the specified exceptions
            if key == 'img_metas' or key == 'gt_masks' or key == 'lane_line' or key == 'lanes':
                data[key] = sample[key]  # Assign the value from the sample dictionary to the data dictionary
                continue

            if key == 'prev_frames':
                data[key] = [to_tensor(np.array(frame)) for frame in sample[key]]
                continue

            data[key] = to_tensor(sample[key])  # Apply the 'to_tensor' function to convert the value to a tensor

        data['img'] = data['img'].permute(2, 0, 1)  # Permute the dimensions of the 'img' tensor

        return data  # Return the modified data dictionary

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'
