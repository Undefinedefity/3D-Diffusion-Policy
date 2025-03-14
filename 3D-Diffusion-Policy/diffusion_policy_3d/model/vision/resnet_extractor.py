import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint
from typing import Dict, Tuple, List
import numpy as np

class SpatialSoftmax(nn.Module):
    """
    空间softmax层，用于保留空间信息
    """
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.data_format = data_format
        
        if temperature:
            self.temperature = nn.Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.0
            
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.width),
            np.linspace(-1., 1., self.height)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)
        
    def forward(self, feature):
        # 输入特征形状: (N, C, H, W)
        if self.data_format == 'NHWC':
            feature = feature.permute(0, 3, 1, 2)
            
        N, C, H, W = feature.shape
        feature = feature.view(N, C, H * W)
        
        # 应用softmax获取空间注意力权重
        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        
        # 计算特征点的期望坐标
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=-1)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=-1)
        
        # 将坐标和特征连接起来
        expected_xy = torch.stack([expected_x, expected_y], dim=-1)
        expected_xy = expected_xy.view(N, C * 2)
        
        return expected_xy

class ResNetEncoder(nn.Module):
    """
    基于ResNet-18的视觉编码器，根据Diffusion Policy论文中的描述进行修改:
    1. 替换全局平均池化为空间softmax池化
    2. 替换BatchNorm为GroupNorm以稳定训练
    """
    def __init__(self, 
                 observation_space: Dict,
                 out_channel: int = 256,
                 img_size: Tuple[int, int] = (224, 224),
                 n_groups: int = 8):
        super().__init__()
        self.rgb_image_key = 'img'
        self.n_output_channels = out_channel
        
        # 检查观察空间中是否包含图像
        if self.rgb_image_key not in observation_space:
            raise ValueError(f"RGB image key '{self.rgb_image_key}' not found in observation space")
        
        # 直接获取shape，因为在hydra配置中它是一个列表
        self.image_shape = observation_space[self.rgb_image_key]
        cprint(f"[ResNetEncoder] image shape: {self.image_shape}", "yellow")
        
        # 检查图像形状是否正确
        if len(self.image_shape) != 3:
            raise ValueError(f"Expected 3D image shape (C,H,W), got {self.image_shape}")
        
        # 创建ResNet-18骨干网络，但替换BatchNorm为GroupNorm
        self.backbone = self._create_modified_resnet18(n_groups)
        
        # 获取ResNet-18最后一层的特征维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.image_shape)  # 使用实际的图像形状
            features = self.backbone(dummy_input)
            _, C, H, W = features.shape
        
        # 添加空间softmax层
        self.spatial_softmax = SpatialSoftmax(height=H, width=W, channel=C)
        
        # 计算空间softmax输出维度并添加投影层
        spatial_softmax_dim = C * 2  # 每个通道有x和y两个坐标
        self.projection = nn.Linear(spatial_softmax_dim, out_channel)
        
        cprint(f"[ResNetEncoder] output dim: {self.n_output_channels}", "red")
    
    def _create_modified_resnet18(self, n_groups):
        """创建修改版的ResNet-18，将BatchNorm替换为GroupNorm"""
        # 导入torchvision中的ResNet
        from torchvision.models.resnet import ResNet, BasicBlock
        
        # 创建自定义的ResNet-18
        model = ResNet(BasicBlock, [2, 2, 2, 2])
        
        # 替换第一层的卷积和归一化层
        model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 替换所有BatchNorm为GroupNorm
        def replace_bn_with_gn(module):
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    num_channels = child.num_features
                    # 确保每组至少有一个通道
                    num_groups = min(n_groups, num_channels)
                    setattr(module, name, nn.GroupNorm(num_groups, num_channels))
                else:
                    replace_bn_with_gn(child)
        
        replace_bn_with_gn(model)
        
        # 移除最后的全连接层和平均池化层
        model = nn.Sequential(*list(model.children())[:-2])
        
        return model
    
    def forward(self, observations: Dict) -> torch.Tensor:
        # 获取图像输入
        images = observations[self.rgb_image_key]
        
        # 确保图像格式正确 [B, C, H, W]
        if len(images.shape) == 4:  # [B, H, W, C]
            images = images.permute(0, 3, 1, 2)  # 转换为[B, C, H, W]
        elif len(images.shape) == 5:  # [B, T, H, W, C]
            B, T, H, W, C = images.shape
            images = images.reshape(B * T, H, W, C)
            images = images.permute(0, 3, 1, 2)  # 转换为[B*T, C, H, W]
        else:
            raise ValueError(f"Unexpected image shape: {images.shape}")
        
        # 通过ResNet骨干网络
        features = self.backbone(images)
        
        # 应用空间softmax
        features = self.spatial_softmax(features)
        
        # 投影到所需的输出维度
        features = self.projection(features)
        
        return features
    
    def output_shape(self):
        return self.n_output_channels 