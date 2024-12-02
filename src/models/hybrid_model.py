import torch
import torch.nn as nn
import torchvision.models as models

class HybridModel(nn.Module):
    def __init__(self, num_classes=10):
        super(HybridModel, self).__init__()
        
        # ResNet50 分支
        self.resnet = models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        num_features_resnet = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # EfficientNet 分支
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        num_features_efficient = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()
        
        # ResNet特征提取
        self.resnet_fc = nn.Sequential(
            nn.Linear(num_features_resnet, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # EfficientNet特征提取
        self.efficient_fc = nn.Sequential(
            nn.Linear(num_features_efficient, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # 多头注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=1024, num_heads=8)
        self.layer_norm = nn.LayerNorm(1024)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x1, x2):
        # ResNet分支
        x1 = self.resnet(x1)
        x1 = self.resnet_fc(x1)
        
        # EfficientNet分支
        x2 = self.efficientnet(x2)
        x2 = self.efficient_fc(x2)
        
        # 特征融合
        combined = torch.cat((x1, x2), dim=1)
        
        # 注意力机制
        attention_input = combined.unsqueeze(0)  # 添加序列维度
        attention_output, _ = self.attention(attention_input, attention_input, attention_input)
        attention_output = attention_output.squeeze(0)  # 移除序列维度
        attention_output = self.layer_norm(attention_output + combined)
        
        # 分类
        output = self.classifier(attention_output)
        return output

def unfreeze_layers(model, num_layers=10):
    """
    解冻模型的最后几层进行微调
    
    Args:
        model: PyTorch模型
        num_layers: 要解冻的层数
    """
    # 解冻ResNet的最后几层
    for i, param in enumerate(reversed(list(model.resnet.parameters()))):
        if i < num_layers:
            param.requires_grad = True
            
    # 解冻EfficientNet的最后几层
    for i, param in enumerate(reversed(list(model.efficientnet.parameters()))):
        if i < num_layers:
            param.requires_grad = True

def build_hybrid_model(num_classes=10, device='cuda'):
    """
    构建混合模型
    
    Args:
        num_classes: 分类数量
        device: 运行设备 ('cuda' 或 'cpu')
    
    Returns:
        model: 构建好的混合模型
    """
    model = HybridModel(num_classes=num_classes)
    model = model.to(device)
    return model

def save_model(model, path, epoch, optimizer, scheduler=None, best_metric=None):
    """
    保存模型检查点
    
    Args:
        model: PyTorch模型
        path: 保存路径
        epoch: 当前轮次
        optimizer: 优化器
        scheduler: 学习率调度器（可选）
        best_metric: 最佳指标（可选）
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    if best_metric is not None:
        checkpoint['best_metric'] = best_metric
        
    torch.save(checkpoint, path)

def load_model(path, model, optimizer=None, scheduler=None, device='cuda'):
    """
    加载模型检查点
    
    Args:
        path: 检查点路径
        model: PyTorch模型
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        device: 运行设备
    
    Returns:
        model: 加载后的模型
        optimizer: 加载后的优化器
        scheduler: 加载后的调度器
        epoch: 加载的轮次
        best_metric: 最佳指标
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', None)
    
    return model, optimizer, scheduler, epoch, best_metric
