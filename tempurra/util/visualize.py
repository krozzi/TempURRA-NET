from typing import List

import cv2
import numpy as np
import torch
import secrets


def display_raw(tensor: torch.Tensor) -> None:
    """
    Visualize a tensor using OpenCV
    :param tensor: The tensor you'd like to visualize. Must be either (B, C, H, W) or (C, H, W).
    :return: None
    """

    for idx, frame in enumerate(tensor):
        frame = frame.detach().cpu().numpy().transpose(1, 2, 0)

        cv2.imshow(f"Image ID#{idx}", frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_multi(tensors: List[torch.Tensor]) -> None:
    """
    Visualize a tensor using OpenCV
    :param tensors: A list of tensors you'd like to visualize. Must be either (B, C, H, W) or (C, H, W).
    :return: None
    """

    for idx, tensor in enumerate(tensors):
        tensors[idx] = tensor.detach().cpu().numpy()

    for tensor in tensors:
        for idx, frame in enumerate(tensor):
            frame = (frame * 255).astype(np.uint8)
            cv2_image = np.transpose(frame, (0, 1)).copy()

            cv2.imshow(f"Image #{secrets.token_hex(64)}", cv2_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


