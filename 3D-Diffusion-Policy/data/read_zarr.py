import os
import zarr
import numpy as np
from termcolor import cprint
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
from PIL import Image

def print_zarr_info(zarr_path):
    """
    读取并打印zarr文件的基本信息
    """
    # 检查文件是否存在
    if not os.path.exists(zarr_path):
        cprint(f"Error: File {zarr_path} does not exist!", "red")
        return

    # 使用ReplayBuffer加载zarr文件
    try:
        buffer = ReplayBuffer.copy_from_path(zarr_path)
        
        # 打印基本信息
        cprint("-" * 50, "cyan")
        cprint("Zarr File Information:", "cyan")
        cprint(f"Path: {zarr_path}", "cyan")
        cprint(f"Number of episodes: {buffer.n_episodes}", "green")
        cprint(f"Total steps: {buffer.n_steps}", "green")
        
        # 打印数据键和对应的数组信息
        cprint("\nData Arrays:", "cyan")
        for key, value in buffer.items():
            cprint(f"\nKey: {key}", "yellow")
            cprint(f"Shape: {value.shape}", "green")
            cprint(f"Dtype: {value.dtype}", "green")
            if isinstance(value, np.ndarray) or isinstance(value, zarr.Array):
                cprint(f"Value range: [{np.min(value):.3f}, {np.max(value):.3f}]", "green")
            
            # 如果是图像数据，打印额外信息
            if key == 'img':
                if len(value.shape) == 4:
                    cprint(f"Image format: {value.shape[1:]} (HxWxC or CxHxW)", "green")
        
        # 打印episode_ends信息
        cprint("\nEpisode Information:", "cyan")
        episode_ends = buffer.episode_ends
        episode_lengths = np.diff(np.concatenate([[0], episode_ends]))
        cprint(f"Episode lengths: min={np.min(episode_lengths)}, max={np.max(episode_lengths)}, "
               f"mean={np.mean(episode_lengths):.1f}", "green")

    except Exception as e:
        cprint(f"Error reading zarr file: {str(e)}", "red")

def read_zarr_images(zarr_path, episode_idx=None, step_idx=None):
    """
    读取zarr文件中的图像数据
    
    参数:
        zarr_path: zarr文件路径
        episode_idx: 指定要读取的episode索引，如果为None则返回所有episodes的图像
        step_idx: 指定要读取的step索引，如果为None则返回指定episode的所有步骤
    
    返回:
        numpy数组格式的图像数据
    """
    try:
        buffer = ReplayBuffer.copy_from_path(zarr_path)
        
        if 'img' not in buffer:
            cprint("Error: No image data found in the zarr file!", "red")
            return None
            
        images = buffer['img']
        
        if episode_idx is not None:
            # 获取指定episode的起始和结束索引
            start_idx = 0 if episode_idx == 0 else buffer.episode_ends[episode_idx - 1]
            end_idx = buffer.episode_ends[episode_idx]
            
            if step_idx is not None:
                # 确保step_idx在有效范围内
                if step_idx >= (end_idx - start_idx):
                    cprint(f"Error: step_idx {step_idx} is out of range!", "red")
                    return None
                return images[start_idx + step_idx]
            
            return images[start_idx:end_idx]
        
        return images
        
    except Exception as e:
        cprint(f"Error reading images from zarr file: {str(e)}", "red")
        return None

def plot_zarr_images(images, num_images=5, figsize=(15, 3)):
    """
    可视化zarr文件中的图像数据
    """
    if images is None:
        cprint("没有图像数据可以显示", "red")
        return
        
    # 打印调试信息
    cprint(f"图像数据类型: {images.dtype}", "yellow")
    cprint(f"图像数据形状: {images.shape}", "yellow")
    cprint(f"图像数据范围: [{images.min()}, {images.max()}]", "yellow")
    
    # 处理单张图像的情况
    if len(images.shape) == 3:
        images = images[np.newaxis, ...]
    
    # 确定要显示的图像数量
    n = min(num_images, len(images))
    
    # 创建图像网格
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    
    for i in range(n):
        img = images[i].copy()  # 创建副本以避免修改原始数据
        
        # 确保图像是HxWxC格式
        if len(img.shape) == 3 and img.shape[-1] != 3:  # 如果不是HxWxC格式
            img = np.transpose(img, (1, 2, 0))
        
        # 数据类型处理
        if img.dtype == np.uint8:
            # uint8类型保持不变
            pass
        else:
            # 其他类型归一化到0-255范围并转换为uint8
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        
        # 显示图像
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Frame {i}')
    
    plt.tight_layout()
    plt.show()

def save_zarr_image(image, save_path):
    """
    保存zarr图像数据到本地文件
    
    参数:
        image: numpy数组格式的图像数据
        save_path: 保存路径，例如 'output.png'
    """
    try:
        # 确保图像是HxWxC格式
        if len(image.shape) == 3 and image.shape[-1] != 3:
            image = np.transpose(image, (1, 2, 0))
            
        # 数据类型处理
        if image.dtype != np.uint8:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            
        # 转换为PIL图像并保存
        img = Image.fromarray(image)
        img.save(save_path)
        cprint(f"图像已保存到: {save_path}", "green")
        
    except Exception as e:
        cprint(f"保存图像时出错: {str(e)}", "red")

def main():
    # 可以修改为你的zarr文件路径
    # zarr_path = "./adroit_hammer_expert.zarr"
    zarr_path = "./metaworld_box-close_expert.zarr"
    
    # 打印文件信息
    print_zarr_info(zarr_path)
    
    # 示例：读取图像数据
    # 读取所有图像
    all_images = read_zarr_images(zarr_path)
    if all_images is not None:
        cprint(f"\n所有图像数据形状: {all_images.shape}", "cyan")
    
    # 读取第一个episode的所有图像
    episode_images = read_zarr_images(zarr_path, episode_idx=0)
    if episode_images is not None:
        cprint(f"第一个episode的图像数据形状: {episode_images.shape}", "cyan")
    
    # 读取第一个episode的第一帧图像
    single_image = read_zarr_images(zarr_path, episode_idx=0, step_idx=0)
    if single_image is not None:
        cprint(f"单帧图像数据形状: {single_image.shape}", "cyan")
        # save the image
        save_zarr_image(single_image, 'zarr_image_frame0.png')
        # plot the image
        plot_zarr_images(single_image)
        # print the image
        print(single_image)

if __name__ == "__main__":
    main()