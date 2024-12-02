import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

class Visualizer:
    """可视化工具类"""
    
    @staticmethod
    def plot_images(images, titles=None, cols=5, figsize=(15, 15)):
        """
        绘制图像网格
        
        Args:
            images: 图像列表
            titles: 标题列表
            cols: 列数
            figsize: 图像大小
        """
        rows = len(images) // cols + (1 if len(images) % cols != 0 else 0)
        fig = plt.figure(figsize=figsize)
        
        for i, image in enumerate(images):
            ax = fig.add_subplot(rows, cols, i + 1)
            if titles is not None:
                ax.set_title(titles[i])
            plt.imshow(image)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_anomaly_detection(original, reconstructed, diff, threshold=0.1):
        """
        绘制异常检测结果
        
        Args:
            original: 原始图像
            reconstructed: 重建图像
            diff: 差异图像
            threshold: 异常阈值
        """
        plt.figure(figsize=(15, 5))
        
        # 原始图像
        plt.subplot(131)
        plt.title('Original')
        plt.imshow(original)
        plt.axis('off')
        
        # 重建图像
        plt.subplot(132)
        plt.title('Reconstructed')
        plt.imshow(reconstructed)
        plt.axis('off')
        
        # 差异图像
        plt.subplot(133)
        plt.title('Anomaly Map')
        anomaly_map = (diff > threshold).astype(np.float32)
        plt.imshow(anomaly_map, cmap='jet')
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def overlay_anomaly_map(image, anomaly_map, alpha=0.5):
        """
        将异常图叠加到原始图像上
        
        Args:
            image: 原始图像
            anomaly_map: 异常图
            alpha: 透明度
            
        Returns:
            overlayed: 叠加后的图像
        """
        # 将异常图转换为热力图
        heatmap = cv2.applyColorMap(np.uint8(255 * anomaly_map), cv2.COLORMAP_JET)
        
        # 将热力图叠加到原始图像上
        overlayed = cv2.addWeighted(np.uint8(image * 255), 1 - alpha,
                                  heatmap, alpha, 0)
        
        return overlayed
    
    @staticmethod
    def plot_training_metrics(metrics_dict, figsize=(12, 4)):
        """
        绘制训练指标
        
        Args:
            metrics_dict: 指标字典，key为指标名称，value为指标值列表
            figsize: 图像大小
        """
        plt.figure(figsize=figsize)
        
        for i, (metric_name, values) in enumerate(metrics_dict.items()):
            plt.subplot(1, len(metrics_dict), i + 1)
            plt.plot(values)
            plt.title(metric_name)
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def save_image(image, path):
        """
        保存图像
        
        Args:
            image: 图像数组
            path: 保存路径
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray((image * 255).astype(np.uint8))
        image.save(path)
