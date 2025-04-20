from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.resnet import resnet50
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.models._utils import IntermediateLayerGetter
from typing import Any, List, Optional, Callable
# from torchvision.models.detection import MaskRCNN
from maskrcnnmodel import MaskRCNN
# from cascadercnnmodel import MaskRCNN
from dropblock import AlwaysDropBlock2d


from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights
from torchvision.ops import misc as misc_nn_ops
import torch.nn.functional as FF
from collections import OrderedDict
from typing import Dict, List, Optional, Callable

from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock
from typing import Dict, List, Optional, Callable
from torch import Tensor
import random
import cv2  # OpenCV 用于混合半透明 mask
from matplotlib.animation import FuncAnimation, PillowWriter
import os


class AlwaysDropout(nn.Module):
    """
    自定义的Dropout层，在训练和评估模式下都保持启用
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p

    def forward(self, x):
        return FF.dropout(x, p=self.p, training=True)  # 始终将training设为True

class AlwaysDropout2d(nn.Module):
    """
    自定义的Dropout2d层，在训练和评估模式下都保持启用
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p

    def forward(self, x):
        return FF.dropout2d(x, p=self.p, training=True)  # 始终将training设为True

class CustomFPN(nn.Module):
    """
    自定义特征金字塔网络实现，dropout在eval模式下也保持启用
    """
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks=None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout_p: float = 0.3
    ):
        super().__init__()
        
        if not in_channels_list:
            raise ValueError("输入通道列表不能为空")
            
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        # self.dropout = AlwaysDropout2d(p=dropout_p)  # 使用自定义的always-on dropout
        self.dropout = AlwaysDropBlock2d(block_size=7,drop_prob=dropout_p)
        
        for in_channels in in_channels_list:
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)
        
        # 如果需要norm_layer，创建norm_blocks
        if norm_layer:
            self.norm_blocks = nn.ModuleList(
                [norm_layer(out_channels) for _ in in_channels_list]
            )
        else:
            self.norm_blocks = None
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        self.extra_blocks = extra_blocks
        self.out_channels = out_channels
        
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向传播过程，在每个特征层后应用always-on dropout"""
        names = list(x.keys())
        results = []
        
        # 获取最后一层特征，应用dropout
        last_inner = self.inner_blocks[-1](x[names[-1]])
        last_inner = self.dropout(last_inner)  # 应用dropout
        
        # 应用最后一层的FPN变换
        last_feature = self.layer_blocks[-1](last_inner)
        if self.norm_blocks:
            last_feature = self.norm_blocks[-1](last_feature)
        last_feature = self.dropout(last_feature)  # 应用dropout
        results.append(last_feature)
        
        # 自顶向下路径
        for idx in range(len(names) - 2, -1, -1):
            # 当前层的lateral connection
            inner_lateral = self.inner_blocks[idx](x[names[idx]])
            inner_lateral = self.dropout(inner_lateral)  # 应用dropout
            
            # 上采样和特征融合
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = FF.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            
            # 应用FPN变换
            layer_feature = self.layer_blocks[idx](last_inner)
            if self.norm_blocks:
                layer_feature = self.norm_blocks[idx](layer_feature)
            layer_feature = self.dropout(layer_feature)  # 应用dropout
            results.insert(0, layer_feature)
        
        # 处理额外的blocks
        if self.extra_blocks is not None:
            if isinstance(self.extra_blocks, LastLevelMaxPool):
                results.append(FF.max_pool2d(results[-1], 1, 2, 0))
                names = list(names) + ["pool"]
            else:
                additional_results = self.extra_blocks(results[-1])
                if additional_results is not None:
                    results.extend(additional_results)
                    names = list(names) + [f"p{len(names) + i}" for i in range(len(additional_results))]
        
        # 构建输出字典
        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        return out
    

class BackboneWithFPN(nn.Module):
    """
    为模型添加特征金字塔网络(FPN)。
    Args:
        backbone (nn.Module): 主干网络模型
        return_layers (Dict[name, new_name]): 包含需要返回激活值的模块名称的字典
        in_channels_list (List[int]): 每个特征图的通道数
        out_channels (int): FPN的输出通道数
        norm_layer (callable, optional): 指定使用的归一化层
    """
    def __init__(
        self,
        backbone: nn.Module,
        return_layers: Dict[str, str],
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = CustomFPN(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.body(x)
        x = self.fpn(x)
        return x

def _custom_resnet_fpn_extractor(
    backbone: nn.Module,
    trainable_layers: int,
    returned_layers: Optional[List[int]] = None,
    extra_blocks: Optional[ExtraFPNBlock] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
) -> BackboneWithFPN:
    """
    构建一个带有FPN的ResNet特征提取器
    
    Args:
        backbone: ResNet主干网络
        trainable_layers: 需要训练的层数
        returned_layers: 需要返回的层
        extra_blocks: 额外的FPN模块
        norm_layer: 归一化层
    """
    # 选择不被冻结的层
    if trainable_layers < 0 or trainable_layers > 5:
        raise ValueError(f"可训练层数应该在[0,5]范围内，得到{trainable_layers}")
    
    # 定义要训练的层
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append("bn1")
    
    # 冻结不需要训练的层
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    # 设置额外的blocks
    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    # 处理返回层
    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    if min(returned_layers) <= 0 or max(returned_layers) >= 5:
        raise ValueError(f"每个返回层应该在[1,4]范围内，得到{returned_layers}")
    
    # 构建返回层字典
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

    # 计算通道数
    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256

    # 构建并返回带FPN的主干网络
    return BackboneWithFPN(
        backbone,
        return_layers,
        in_channels_list,
        out_channels,
        extra_blocks=extra_blocks,
        norm_layer=norm_layer
    )

class IntermediateLayerGetter(nn.ModuleDict):
    """
    模块封装器，用于获取中间层的输出
    
    Args:
        model (nn.Module): 要包装的模型
        return_layers (Dict[name, new_name]): 需要获取输出的层名字典
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers参数中指定的键不是model的子模块")
        
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        
        # 重新构建模型，只保留到最后需要的层
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out
        

    
def custom_maskrcnn_resnet50_fpn(
    *,
    weights: Optional[MaskRCNN_ResNet50_FPN_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
) -> MaskRCNN:
    """
    自定义实现的 MaskRCNN-ResNet50-FPN 模型。
    """
    # 验证权重参数
    weights = MaskRCNN_ResNet50_FPN_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)

    # 如果有预训练的 COCO 权重，自动设置类别数
    if weights is not None:
        weights_backbone = None
        num_classes = num_classes or len(weights.meta["categories"])
    elif num_classes is None:
        num_classes = 91  # 默认类别数（COCO）

    # 验证主干网络中可训练层数
    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)

    # 使用 FrozenBatchNorm2d（冻结的归一化层）或标准 BatchNorm2d
    norm_layer = misc_nn_ops.FrozenBatchNorm2d if is_trained else nn.BatchNorm2d

    # 创建 ResNet-50 主干网络
    backbone = resnet50(weights=weights_backbone, progress=progress, norm_layer=norm_layer)

    # 将 ResNet-50 转换为带 FPN 的主干网络
    # backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    backbone = _custom_resnet_fpn_extractor(backbone, trainable_backbone_layers)


    # 创建 MaskRCNN 模型
    model = MaskRCNN(backbone, num_classes=num_classes, **kwargs)

    # 加载预训练的 COCO 权重（如果指定）
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress),strict=False)
        # if weights == MaskRCNN_ResNet50_FPN_Weights.COCO_V1:
        #     overwrite_eps(model, 0.0)

    return model

