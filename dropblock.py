import torch
import torch.nn as nn
import torch.nn.functional as FF

# class DropBlock2D(nn.Module):
#     """
#     DropBlock2D: 丢弃连续的特征块
#     参考: https://arxiv.org/pdf/1810.12890.pdf
#     """
#     def __init__(self, block_size: int = 7, drop_prob: float = 0.1):
#         """
#         初始化 DropBlock2D
#         :param block_size: 要丢弃的特征块大小 (默认: 7)
#         :param drop_prob: 丢弃概率 (默认: 0.1)
#         """
#         super(DropBlock2D, self).__init__()
#         self.block_size = block_size
#         self.drop_prob = drop_prob

#     def forward(self, x):
#         """
#         前向传播，训练模式下应用 DropBlock2D
#         :param x: 输入特征图 [B, C, H, W]
#         """
#         if not self.training or self.drop_prob == 0.0:
#             # 如果处于评估模式或 drop_prob 为 0，则直接返回输入
#             return x

#         # 获取输入形状
#         gamma = self._compute_gamma(x)
#         batch_size, channels, height, width = x.shape

#         # 生成随机掩码
#         mask = (torch.rand(batch_size, channels, height, width, device=x.device) < gamma).float()

#         # 使用最大池化扩展掩码为 block_size 的块
#         mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
#         mask = 1 - mask  # 反转掩码，丢弃的区域为 0

#         # 重新缩放输入，保持特征图的分布一致
#         mask_area = mask.numel() / torch.sum(mask)
#         return x * mask * mask_area

#     def _compute_gamma(self, x):
#         """
#         计算 gamma，用于生成随机掩码
#         :param x: 输入特征图
#         :return: gamma 值
#         """
#         height, width = x.shape[2], x.shape[3]
#         return self.drop_prob / (self.block_size ** 2) * (height * width) / ((height - self.block_size + 1) * (width - self.block_size + 1))


# class AlwaysDropBlock2D(nn.Module):
#     """
#     始终启用 DropBlock2D，在训练和评估模式下都启用
#     """
#     def __init__(self, block_size: int = 7, drop_prob: float = 0.1):
#         """
#         初始化 AlwaysDropBlock2D
#         :param block_size: 要丢弃的特征块大小 (默认: 7)
#         :param drop_prob: 丢弃概率 (默认: 0.1)
#         """
#         super(AlwaysDropBlock2D, self).__init__()
#         self.dropblock = DropBlock2D(block_size=block_size, drop_prob=drop_prob)

#     def forward(self, x):
#         # 始终强制启用 DropBlock2D
#         return self.dropblock(x)


class AlwaysDropBlock2d(nn.Module):
    """
    自定义的 DropBlock2d 层，在训练和评估模式下都保持启用
    """
    def __init__(self, block_size: int = 7, drop_prob: float = 0.1):
        """
        初始化 AlwaysDropBlock2d
        :param block_size: 丢弃的特征块大小
        :param drop_prob: 丢弃概率
        """
        super(AlwaysDropBlock2d, self).__init__()
        if drop_prob < 0 or drop_prob > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {drop_prob}")
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0:
            return x  # 如果丢弃概率为 0，则直接返回输入
        
        # 获取输入特征图大小
        gamma = self._compute_gamma(x)
        batch_size, channels, height, width = x.shape

        # 生成随机掩码
        mask = (torch.rand(batch_size, channels, height, width, device=x.device) < gamma).float()

        # 扩展掩码为 block_size 的连续块
        mask = FF.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
        mask = 1 - mask  # 反转掩码，丢弃区域为 0

        # 重新缩放输入
        mask_area = mask.numel() / torch.sum(mask)
        return x * mask * mask_area

    def _compute_gamma(self, x):
        """
        计算 gamma，用于生成随机掩码
        """
        height, width = x.shape[2], x.shape[3]
        return self.drop_prob / (self.block_size ** 2) * (height * width) / ((height - self.block_size + 1) * (width - self.block_size + 1))