import torch
import torch.nn as nn
from typing import Dict
from termcolor import cprint

from diffusion_policy_3d.model.vision.pointnet_extractor import DP3Encoder
from diffusion_policy_3d.model.vision.resnet_extractor import ResNetEncoder

class FusionEncoder(nn.Module):
    """
    融合点云和RGB图像的编码器
    """
    def __init__(self, 
                 observation_space: Dict,
                 img_crop_shape=None,
                 out_channel=256,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type="pointnet",
                 use_rgb_image=True,
                 image_encoder_out_dim=128,
                 img_size=(224, 224),
                 n_groups=8):
        super().__init__()
        
        self.use_rgb_image = use_rgb_image
        self.rgb_image_key = 'img'
        self.point_cloud_key = 'point_cloud'
        
        # 检查观察空间中是否包含必要的键
        has_point_cloud = self.point_cloud_key in observation_space
        has_rgb_image = self.rgb_image_key in observation_space if use_rgb_image else False
        
        if not has_point_cloud:
            raise ValueError(f"Point cloud key '{self.point_cloud_key}' not found in observation space")
        
        if use_rgb_image and not has_rgb_image:
            raise ValueError(f"RGB image key '{self.rgb_image_key}' not found in observation space")
        
        # 创建点云编码器
        self.point_cloud_encoder = DP3Encoder(
            observation_space=observation_space,
            img_crop_shape=img_crop_shape,
            out_channel=out_channel,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type
        )
        
        # 如果使用RGB图像，创建图像编码器
        if use_rgb_image:
            self.image_encoder = ResNetEncoder(
                observation_space=observation_space,
                out_channel=image_encoder_out_dim,
                img_size=img_size,
                n_groups=n_groups
            )
            # 总输出维度是点云编码器和图像编码器的输出维度之和
            self.n_output_channels = self.point_cloud_encoder.output_shape() + image_encoder_out_dim
        else:
            self.image_encoder = None
            self.n_output_channels = self.point_cloud_encoder.output_shape()
        
        cprint(f"[FusionEncoder] use_rgb_image: {use_rgb_image}", "yellow")
        cprint(f"[FusionEncoder] output dim: {self.n_output_channels}", "red")
    
    def forward(self, observations: Dict) -> torch.Tensor:
        # 获取点云特征
        point_cloud_features = self.point_cloud_encoder(observations)
        
        # 如果使用RGB图像，获取图像特征并拼接
        if self.use_rgb_image and self.image_encoder is not None:
            image_features = self.image_encoder(observations)
            features = torch.cat([point_cloud_features, image_features], dim=-1)
        else:
            features = point_cloud_features
        
        return features
    
    def output_shape(self):
        return self.n_output_channels 