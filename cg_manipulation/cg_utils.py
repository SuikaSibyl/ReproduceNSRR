import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from typing import Union, Tuple, List
import sys

def zero_upsampling(input: torch.Tensor,
                    scale_factor: Tuple[int, int]) -> torch.Tensor:
    """
    Do zero upsampling for 4D input
    """
    # input shape is: batch x channels x height x width
    input_size = torch.tensor(input.size(), dtype=torch.int)
    # Get the last two dimensions -> height x width
    input_image_size = input_size[2:]
    data_size = input_size[:2]

    # Calc output shape
    scale_factor = torch.from_numpy(np.asarray(scale_factor)).int()
    # Ensure the pixel is about to be the center
    offset_factor = scale_factor/2
    # check that the dimensions of the tuples match.
    if len(input_image_size) != len(scale_factor):
        raise ValueError("scale_factor should match input size!")
    output_image_size = (input_image_size * scale_factor).type(torch.int)

    # Create Output
    output_size = torch.cat((data_size, output_image_size))
    output = torch.zeros(tuple(output_size.tolist()), device = input.device)
    output[:, :, offset_factor[0]::scale_factor[0], offset_factor[1]::scale_factor[1]] = input
    return output


def backward_warping(img: torch.Tensor,
                         motion: torch.Tensor) -> torch.Tensor:
    """
    Do backward warping for 4D input
    """
    # input shape is: batch x channels x height x width
    index_batch, number_channels, height, width = img.size()
    grid_x = torch.arange(width).view(1, -1).repeat(height, 1)
    grid_y = torch.arange(height).view(-1, 1).repeat(1, width)
    grid_x = grid_x.view(1, 1, height, width).repeat(index_batch, 1, 1, 1)
    grid_y = grid_y.view(1, 1, height, width).repeat(index_batch, 1, 1, 1)
    ##
    grid = torch.cat((grid_x, grid_y), 1).cuda().float()
    # grid is: [batch, channel (2), height, width]
    vgrid = grid + motion
    # Grid values must be normalised positions in [-1, 1]
    vgrid_x = vgrid[:, 0, :, :]
    vgrid_y = vgrid[:, 1, :, :]
    vgrid[:, 0, :, :] = (vgrid_x / width) * 2.0 - 1.0
    vgrid[:, 1, :, :] = (vgrid_y / height) * 2.0 - 1.0
    # swapping grid dimensions in order to match the input of grid_sample.
    # that is: [batch, output_height, output_width, grid_pos (2)]
    vgrid = vgrid.permute((0, 2, 3, 1))
    output = F.grid_sample(img, vgrid, mode='bilinear')
    return output

def rgb_to_hsv(img):
    eps = 1e-8
    hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

    hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 2] == img.max(1)[0]]
    hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 1] == img.max(1)[0]]
    hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 0] == img.max(1)[0]]) % 6

    hue[img.min(1)[0] == img.max(1)[0]] = 0.0
    hue = hue / 6

    saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + eps)
    saturation[img.max(1)[0] == 0] = 0

    value = img.max(1)[0]

    hue = hue.unsqueeze(1)
    saturation = saturation.unsqueeze(1)
    value = value.unsqueeze(1)
    hsv = torch.cat([hue, saturation, value], dim=1)
    return hsv


def optical_flow_to_motion(rgb_flow: torch.Tensor, sensitivity: float = 0.5) -> torch.Tensor:
    """
    Returns motion vectors as a [batch, 2, height, width]
    with [:, 0, :, :] the abscissa and [:, 1, :, :] the ordinate.
    """
    # flow is: batch x 3-channel x height x width
    hsv_flow = rgb_to_hsv(rgb_flow)
    motion_length = hsv_flow[:, 2, :, :] / sensitivity
    motion_angle = (hsv_flow[:, 0, :, :] - 0.5) * (2.0 * np.pi)
    motion_x = motion_length * torch.cos(motion_angle) * rgb_flow.shape[3] * 2
    motion_y = - motion_length * torch.sin(motion_angle) * rgb_flow.shape[2] * 2
    motion_x.unsqueeze_(1)
    motion_y.unsqueeze_(1)
    # motion is: batch x 2-channel x height x width
    motion = torch.cat((motion_x, motion_y), dim=1)
    return motion
