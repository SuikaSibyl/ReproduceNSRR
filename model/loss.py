import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision
import pytorch_ssim

from model.model import LayerOutputModelDecorator
from threading import Lock, Thread

from typing import List

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

class SingletonPattern(type):
    """
    see: https://refactoring.guru/fr/design-patterns/singleton/python/example
    """
    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]



def nll_loss(output, target):
    return F.nll_loss(output, target)


def feature_reconstruction_loss(conv_layer_output: torch.Tensor, conv_layer_target: torch.Tensor) -> torch.Tensor:
    """
    Computes Feature Reconstruction Loss as defined in Johnson et al. (2016)
    todo: syntax
    Justin Johnson, Alexandre Alahi, and Li Fei-Fei. 2016. Perceptual losses for real-time
    style transfer and super-resolution. In European Conference on Computer Vision.
    694–711.
    Takes the already-computed output from the VGG16 convolution layers.
    """
    if conv_layer_output.shape != conv_layer_target:
        raise ValueError("Output and target tensors have different dimensions!")
    loss = conv_layer_output.dist(conv_layer_target, p=2) / torch.numel(conv_layer_output)
    return loss


def nsrr_loss(output: torch.Tensor, target: torch.Tensor, w: float) -> torch.Tensor:
    """
    Computes the loss as defined in the NSRR paper.
    """
    loss_ssim = 1 - pytorch_ssim.ssim(output, target)
    l1_loss = nn.L1Loss()
    loss_l1 = l1_loss(output, target)
    # loss_ssim_r = 1 - pytorch_ssim.ssim(output[:,0:1,:,:], target[:,0:1,:,:])
    # NSRR_SSIM(target, target)
    # loss_perception = 0
    # conv_layers_output = PerceptualLossManager().get_vgg16_conv_layers_output(output)
    # conv_layers_target = PerceptualLossManager().get_vgg16_conv_layers_output(output)
    # for i in range(len(conv_layers_output)):
    #     loss_perception += feature_reconstruction_loss(conv_layers_output[i], conv_layers_target[i])
    loss = loss_ssim #+ w * loss_perception
    return loss


class PerceptualLossManager(metaclass=SingletonPattern):
    """
    Singleton
    """
    # Init
    def __init__(self):
        self.vgg_model = torchvision.models.vgg16(pretrained=True, progress=True)
        self.vgg_model.eval()
        """ 
            Feature Reconstruction Loss 
            - needs output from each convolution layer.
        """
        self.layer_predicate = lambda name, module: type(module) == nn.Conv2d
        self.lom = LayerOutputModelDecorator(self.vgg_model.features, self.layer_predicate).to(device=torch.device('cuda:0'))

    def get_vgg16_conv_layers_output(self, x: torch.Tensor)-> List[torch.Tensor]:
        """
        Returns the list of output of x on the pre-trained VGG16 model for each convolution layer.
        """
        return self.lom.forward(x)


def NSRR_SSIM(X, Y):
    # X: (N,3,H,W) a batch of non-negative RGB images (0~255)
    # Y: (N,3,H,W)

    # calculate ssim & ms-ssim for each image
    ssim_val = ssim(X, Y, data_range=255, size_average=False)  # return (N,)
    ms_ssim_val = ms_ssim(X, Y, data_range=255, size_average=False)  # (N,)

    # set 'size_average=True' to get a scalar value as loss. see tests/tests_loss.py for more details
    ssim_loss = 1 - ssim(X, Y, data_range=255, size_average=True)  # return a scalar
    ms_ssim_loss = 1 - ms_ssim(X, Y, data_range=255, size_average=True)

    # reuse the gaussian kernel with SSIM & MS_SSIM.
    ssim_module = SSIM(data_range=255, size_average=True, channel=3)
    ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)

    ssim_loss = 1 - ssim_module(X, Y)
    ms_ssim_loss = 1 - ms_ssim_module(X, Y)