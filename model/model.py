import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from cg_manipulation.cg_utils import zero_upsampling, backward_warping, optical_flow_to_motion
from typing import Union, List, Tuple, Callable, Any
import time
import matplotlib.pyplot as plt

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# class NeuralSuperSampling(BaseModel):
#     def __init__(self):
#         super().__init__()
#
#         scale = 10
#         kernel_size = 3
#         # Adding padding here so that we do not lose width or height because of the convolutions.
#         # The input and output must have the same image dimensions so that we may concatenate them
#         padding = 1
#
#         self.FeatureExtraction = nn.Sequential(
#             nn.Conv2d(4, 32, kernel_size=kernel_size, padding=padding),
#             nn.ReLU(True),
#             nn.Conv2d(32, 32, kernel_size=kernel_size, padding=padding),
#             nn.ReLU(True),
#             nn.Conv2d(32, 8, kernel_size=kernel_size, padding=padding),
#             nn.ReLU(True)
#         )
#
#         self.FeatureReweighting = nn.Sequential(
#             nn.Conv2d(4, 32, kernel_size=kernel_size, padding=padding),
#             nn.ReLU(True),
#             nn.Conv2d(32, 32, kernel_size=kernel_size, padding=padding),
#             nn.ReLU(True),
#             nn.Conv2d(32, 4, kernel_size=kernel_size, padding=padding),
#             nn.Tanh(),
#             Remap([-1, 1], [0, scale])
#         )
#
#     def forward(self, frame, prev1, prev2, prev3, prev4):
#         frame_feature = self.FeatureExtraction(frame)
#
#         prev1_feature = self.FeatureExtraction(prev1)
#         prev1_upsamp = ZeroUpsampling(prev1_feature)
#         prev1_wraped = BackwardWraping(prev1_upsamp)
#
#         prev2_feature = self.FeatureExtraction(prev2)
#         prev2_upsamp = ZeroUpsampling(prev2_feature)
#         prev2_wraped = BackwardWraping(prev2_upsamp)
#
#         prev3_feature = self.FeatureExtraction(prev3)
#         prev3_upsamp = ZeroUpsampling(prev3_feature)
#         prev3_wraped = BackwardWraping(prev3_upsamp)
#
#         prev4_feature = self.FeatureExtraction(prev4)
#         prev4_upsamp = ZeroUpsampling(prev4_feature)
#         prev4_wraped = BackwardWraping(prev4_upsamp)
#
#         rgbd_all =
#         return x1


class FeatureExtraction(BaseModel):
    def __init__(self):
        super().__init__()

        kernel_size = 3
        # Adding padding here so that we do not lose width or height because of the convolutions.
        # The input and output must have the same image dimensions so that we may concatenate them
        padding = 1

        self.FeatureExtraction = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(True),
            nn.Conv2d(32, 8, kernel_size=kernel_size, padding=padding),
            nn.ReLU(True),
        )

    def forward(self, x):
        feature = self.FeatureExtraction(x)
        x = torch.cat((x, feature), 1)
        return x


class ZeroUpsampling(BaseModel):
    """
    Basic layer for zero-upsampling of 2D images (4D tensors).
    """

    scale_factor: Tuple[int, int]

    def __init__(self, scale_factor: Tuple[int, int]):
        super(ZeroUpsampling, self).__init__()
        assert(len(scale_factor) == 2)
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return zero_upsampling(x, scale_factor=self.scale_factor)


# class BackwardWarp(BaseModel):
#     """
#     A model for backward warping 2D image tensors according to motion tensors.
#     """
#
#     def __init__(self):
#         super(BackwardWarp, self).__init__()
#
#     def forward(self, x_image: torch.Tensor, x_motion: torch.Tensor) -> torch.Tensor:
#         return backward_warp_motion(x_image, x_motion)
#


class Remap(BaseModel):
    """
    Basic layer for element-wise remapping of values from one range to another.
    """

    in_range: Tuple[float, float]
    out_range: Tuple[float, float]

    def __init__(self,
                 in_range: Union[Tuple[float, float], List[float]],
                 out_range: Union[Tuple[float, float], List[float]]
                 ):
        assert(len(in_range) == len(out_range) and len(in_range) == 2)
        super(BaseModel, self).__init__()
        self.in_range = tuple(in_range)
        self.out_range = tuple(out_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.div(
            torch.mul(torch.add(x, - self.in_range[0]), self.out_range[1] - self.out_range[0]),
            (self.in_range[1] - self.in_range[0]) + self.out_range[0])


class ReconstructionNet(BaseModel):
    """
    Reconstruction Network
    Input:  reweighted upsampled previous features
            zero-upsampled current feature
    Input n channels and output 3 channels
    """
    def __init__(self):
        super().__init__()

        kernel_size = 3
        padding = 1

        self.encoder1 = nn.Sequential(
            nn.Conv2d(60, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(True),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(True),
        )
        self.center = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=kernel_size, padding=padding),
            nn.ReLU(True)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(True)
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(True),
            nn.Conv2d(32, 3, kernel_size=kernel_size, padding=padding),
            nn.ReLU(True),
        )

        self.pooling = nn.MaxPool2d(2)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        enocded1 = self.encoder1(x)
        # x = self.pooling(enocded1)
        enocded2 = self.encoder2(enocded1)
        encoded3 = self.pooling(enocded2)
        encoded4 = self.center(encoded3)
        encoded5 = self.upsampling(encoded4)
        encoded6 = torch.cat([encoded5, enocded2], dim=1)
        decoded1 = self.decoder2(encoded6)
        decoded3 = self.decoder1(decoded1)
        return decoded3



class FeatureReweightingNetwork(BaseModel):
    """
    Feature Reweighting Network based on
    "Neural Supersampling for Real-time Rendering", Lei Xiao, ACM SIGGRAPH 2020.
    https://research.fb.com/blog/2020/07/introducing-neural-supersampling-for-real-time-rendering/
    """
    def __init__(self):
        super().__init__()

        kernel_size = 3
        padding = 1
        scale = 10

        self.FeatureReweighting = nn.Sequential(
            nn.Conv2d(20, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(True),
            nn.Conv2d(32, 4, kernel_size=kernel_size, padding=padding),
            nn.Tanh(),
            Remap([-1, 1], [0, scale])
        )

    def forward(self, feature_0, feature_1, feature_2, feature_3, feature_4):
        rgbd_0 = feature_0[:,0:4,:,:]
        rgbd_1 = feature_1[:,0:4,:,:]
        rgbd_2 = feature_2[:,0:4,:,:]
        rgbd_3 = feature_3[:,0:4,:,:]
        rgbd_4 = feature_3[:,0:4,:,:]

        input = (torch.cat((rgbd_0, rgbd_1, rgbd_2, rgbd_3, rgbd_4), 1))
        weight = self.FeatureReweighting(input)

        weighted_feature_01 = feature_1 * weight[:,0:1,:,:]
        weighted_feature_02 = feature_2 * weight[:,1:2,:,:]
        weighted_feature_03 = feature_3 * weight[:,2:3,:,:]
        weighted_feature_04 = feature_4 * weight[:,3:4,:,:]

        output = (torch.cat((weighted_feature_01, weighted_feature_02, weighted_feature_03, weighted_feature_04), 1))
        return output


class SuperSamplingNet(BaseModel):
    """
    Neural Super Sampling method based on
    "Neural Supersampling for Real-time Rendering", Lei Xiao, ACM SIGGRAPH 2020.
    https://research.fb.com/blog/2020/07/introducing-neural-supersampling-for-real-time-rendering/
    """
    def __init__(self):
        super().__init__()

        kernel_size = 3
        padding = 1

        self.featureExtraction_curr = FeatureExtraction()
        self.featureExtraction_prev = FeatureExtraction()
        self.zeroUpsampling = ZeroUpsampling([2,2])
        self.recontruction = ReconstructionNet()
        self.upsampling = nn.UpsamplingNearest2d(scale_factor=2)
        self.FeatureReweight = FeatureReweightingNetwork()
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)


    def forward(self, view_0, depth_0, flow_0, view_1, depth_1, flow_1,
                      view_2, depth_2, flow_2, view_3, depth_3, flow_3,
                      view_4, depth_4, flow_4):
        # measure process time

        with torch.no_grad():
            rgbd_curr = (torch.cat((view_0, depth_0), 1))
            rgbd_1 =  (torch.cat((view_1, depth_1), 1))
            rgbd_2 =  (torch.cat((view_2, depth_2), 1))
            rgbd_3 =  (torch.cat((view_3, depth_3), 1))
            rgbd_4 =  (torch.cat((view_4, depth_4), 1))

            flow_0_origin = flow_0
            flow_1_origin = flow_1
            flow_0 = self.upsampling(flow_0)
            flow_1 = self.upsampling(flow_1)
            flow_2 = self.upsampling(flow_2)
            flow_3 = self.upsampling(flow_3)

        t0 = time.clock()
        # Feature Extraction stage
        feature_curr = self.featureExtraction_curr(rgbd_curr)
        feature_1 = self.featureExtraction_prev(rgbd_1)
        feature_2 = self.featureExtraction_prev(rgbd_2)
        feature_3 = self.featureExtraction_prev(rgbd_3)
        feature_4 = self.featureExtraction_prev(rgbd_4)

        # with torch.no_grad():
        #     test_img_1 = feature_curr.cpu()[0].permute(1, 2, 0)[:,:,3:6]
        #     plt.imshow(test_img_1)
        #     plt.show()

        # Zero Upsampling stage
        upsampled_curr = self.zeroUpsampling(feature_curr)
        upsampled_1 = self.zeroUpsampling(feature_1)
        upsampled_2 = self.zeroUpsampling(feature_2)
        upsampled_3 = self.zeroUpsampling(feature_3)
        upsampled_4 = self.zeroUpsampling(feature_4)

        # Backward wraping stage

        feature_1_wraped = backward_warping(upsampled_1, flow_0)

        feature_2_wraped = backward_warping(upsampled_2, flow_1)
        feature_2_wraped = backward_warping(feature_2_wraped, flow_0)

        feature_3_wraped = backward_warping(upsampled_3, flow_2)
        feature_3_wraped = backward_warping(feature_3_wraped, flow_1)
        feature_3_wraped = backward_warping(feature_3_wraped, flow_0)

        feature_4_wraped = backward_warping(upsampled_4, flow_3)
        feature_4_wraped = backward_warping(feature_4_wraped, flow_2)
        feature_4_wraped = backward_warping(feature_4_wraped, flow_1)
        feature_4_wraped = backward_warping(feature_4_wraped, flow_0)

        # Reweighted stage
        reweighted_features = self.FeatureReweight(upsampled_curr, feature_1_wraped, feature_2_wraped, feature_3_wraped, feature_4_wraped)
        input = (torch.cat((upsampled_curr, reweighted_features), 1))

        # Reconstruction stage
        result = self.recontruction(input)
        print (time.clock() - t0, "seconds process time")
        return result


class LayerOutputModelDecorator(BaseModel):
    """
    A Decorator for a Model to output the output from an arbitrary set of layers.
    """

    def __init__(self, model: nn.Module, layer_predicate: Callable[[str, nn.Module], bool]):
        super(LayerOutputModelDecorator, self).__init__()
        self.model = model
        self.layer_predicate = layer_predicate

        self.output_layers = []

        def _layer_forward_func(layer_index: int) -> Callable[[nn.Module, Any, Any], None]:
            def _layer_hook(module_: nn.Module, input_, output) -> None:
                self.output_layers[layer_index] = output

            return _layer_hook

        self.layer_forward_func = _layer_forward_func

        for name, module in self.model.named_children():
            if self.layer_predicate(name, module):
                module.register_forward_hook(
                    self.layer_forward_func(len(self.output_layers)))
                self.output_layers.append(torch.Tensor())

    def forward(self, x) -> List[torch.Tensor]:
        self.model(x)
        return self.output_layers