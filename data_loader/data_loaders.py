from torchvision import datasets, transforms
from base import BaseDataLoader

import os
from base import BaseDataLoader
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as tf

from PIL import Image

from typing import Union, Tuple, List
import torch
import numpy as np


def get_downscaled_size(x: torch.Tensor, downscale_factor: Tuple) -> Tuple:
    """
    Computes (h, w) size of a 4D or (d, h, w) of 5D tensor of images downscaled by a scaling factor (tuple of int).
    """
    return tuple((np.asarray(x.size()[2:]) / np.asarray(downscale_factor)).astype(int).tolist())


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)



class NSRRDataLoader(BaseDataLoader):
    """
    Super Sampling data loading demo using BaseDataLoader
    """
    def __init__(self,
                 root_dir: str,
                 view_dirname: str,
                 depth_dirname: str,
                 flow_dirname: str,
                 batch_size: int,
                 shuffle: bool = True,
                 validation_split: float = 0.0,
                 num_workers: int = 1,
                 downscale_factor: Union[Tuple[int, int], List[int], int] = (2, 2)
                 ):
        self.dataset = NSRRDataset(root_dir,
                              view_dirname=view_dirname,
                              depth_dirname=depth_dirname,
                              flow_dirname=flow_dirname,
                              downscale_factor=downscale_factor
                              )
        super(NSRRDataLoader, self).__init__(dataset=self.dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             validation_split=validation_split,
                                             num_workers=num_workers,
                                             )


class NSRRDataset(Dataset):
    """
    Custom Dataset for Super Sampling Network
    Requires that corresponding view, depth and motion frames share the same name.
    """
    def __init__(self,
                 root_dir: str,
                 view_dirname: str,
                 depth_dirname: str,
                 flow_dirname: str,
                 downscale_factor: Union[Tuple[int, int], List[int], int] = (2, 2),
                 transform: nn.Module = None,
                 ):
        super(NSRRDataset, self).__init__()

        self.root_dir = root_dir
        self.view_dirname = view_dirname
        self.depth_dirname = depth_dirname
        self.flow_dirname = flow_dirname

        self.truth_dirname = "Truth/"

        self.view0_dirname = "View-0/"
        self.view1_dirname = "View-1/"
        self.view2_dirname = "View-2/"
        self.view3_dirname = "View-3/"
        self.view4_dirname = "View-4/"

        self.depth0_dirname = "Depth-0/"
        self.depth1_dirname = "Depth-1/"
        self.depth2_dirname = "Depth-2/"
        self.depth3_dirname = "Depth-3/"
        self.depth4_dirname = "Depth-4/"

        self.motion0_dirname = "Motion-0/"
        self.motion1_dirname = "Motion-1/"
        self.motion2_dirname = "Motion-2/"
        self.motion3_dirname = "Motion-3/"
        self.motion4_dirname = "Motion-4/"

        if type(downscale_factor) == int:
            downscale_factor = (downscale_factor, downscale_factor)
        self.downscale_factor = tuple(downscale_factor)

        if transform is None:
            self.transform = tf.ToTensor()
        self.view_listdir = os.listdir(os.path.join(self.root_dir, self.truth_dirname))

    def __len__(self) -> int:
        return len(self.view_listdir)

    def __getitem__(self, index):
        # view
        image_name = self.view_listdir[index]

        truth_path = os.path.join(self.root_dir, self.truth_dirname, image_name)

        view0_path = os.path.join(self.root_dir, self.view0_dirname, image_name)
        view1_path = os.path.join(self.root_dir, self.view1_dirname, image_name)
        view2_path = os.path.join(self.root_dir, self.view2_dirname, image_name)
        view3_path = os.path.join(self.root_dir, self.view3_dirname, image_name)
        view4_path = os.path.join(self.root_dir, self.view4_dirname, image_name)

        depth0_path = os.path.join(self.root_dir, self.depth0_dirname, image_name)
        depth1_path = os.path.join(self.root_dir, self.depth1_dirname, image_name)
        depth2_path = os.path.join(self.root_dir, self.depth2_dirname, image_name)
        depth3_path = os.path.join(self.root_dir, self.depth3_dirname, image_name)
        depth4_path = os.path.join(self.root_dir, self.depth4_dirname, image_name)

        motion0_path = os.path.join(self.root_dir, self.motion0_dirname, image_name)
        motion1_path = os.path.join(self.root_dir, self.motion1_dirname, image_name)
        motion2_path = os.path.join(self.root_dir, self.motion2_dirname, image_name)
        motion3_path = os.path.join(self.root_dir, self.motion3_dirname, image_name)
        motion4_path = os.path.join(self.root_dir, self.motion4_dirname, image_name)

        trans = self.transform

        img_view_truth = trans(Image.open(truth_path))

        img_view_0 = trans(Image.open(view0_path))
        img_view_1 = trans(Image.open(view1_path))
        img_view_2 = trans(Image.open(view2_path))
        img_view_3 = trans(Image.open(view3_path))
        img_view_4 = trans(Image.open(view4_path))

        img_depth_0 = trans(Image.open(depth0_path).convert(mode="L"))
        img_depth_1 = trans(Image.open(depth1_path).convert(mode="L"))
        img_depth_2 = trans(Image.open(depth2_path).convert(mode="L"))
        img_depth_3 = trans(Image.open(depth3_path).convert(mode="L"))
        img_depth_4 = trans(Image.open(depth4_path).convert(mode="L"))

        img_motion_0 = trans(Image.open(motion0_path))
        img_motion_1 = trans(Image.open(motion1_path))
        img_motion_2 = trans(Image.open(motion2_path))
        img_motion_3 = trans(Image.open(motion3_path))
        img_motion_4 = trans(Image.open(motion4_path))

        return img_view_0, img_view_1, img_view_2, img_view_3, img_view_4, \
               img_depth_0, img_depth_1, img_depth_2, img_depth_3, img_depth_4, \
               img_motion_0, img_motion_1, img_motion_2, img_motion_3, img_motion_4, img_view_truth
