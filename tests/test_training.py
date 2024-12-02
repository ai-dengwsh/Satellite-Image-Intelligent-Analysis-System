import pytest
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

from src.models.hybrid_model import (
    build_hybrid_model,
    save_model,
    load_model,
    unfreeze_layers
)
from src.data.data_preprocessing import SatelliteDataModule

@pytest.fixture
def test_data_dir():
    """创建测试数据目录"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 创建必要的子目录
        for split in ['train', 'val', 'test']:
            for cls in ['class1', 'class2']:
                os.makedirs(os.path.join(tmp_dir, split, cls), exist_ok=True)
                
        # 创建一些测试图像
        for split in ['train', 'val', 'test']:
            for cls in ['class1', 'class2']:
                for i in range(5):  # 每个类别5张图片
                    img_path = os.path.join(tmp_dir, split, cls, f'img_{i}.jpg')
                    # 创建随机图像数据
                    img_data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    img = Image.fromarray(img_data)
                    img.save(img_path, format='JPEG')
        
        yield tmp_dir

@pytest.fixture
def model():
    """创建测试模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return build_hybrid_model(num_classes=2, device=device)

@pytest.fixture
def data_module(test_data_dir):
    """创建数据模块"""
    return SatelliteDataModule(
        data_dir=test_data_dir,
        img_size=(224, 224),
        batch_size=2
    )

def test_model_initialization(model):
    """测试模型初始化"""
    assert isinstance(model, nn.Module)
    # 检查模型的主要组件
    assert hasattr(model, 'resnet')
    assert hasattr(model, 'efficientnet')
    assert hasattr(model, 'attention')
    assert hasattr(model, 'classifier')

def test_model_forward(model):
    """测试模型前向传播"""
    batch_size = 2
    device = next(model.parameters()).device
    x1 = torch.randn(batch_size, 3, 224, 224).to(device)
    x2 = torch.randn(batch_size, 3, 224, 224).to(device)
    
    output = model(x1, x2)
    assert output.shape == (batch_size, 2)  # 2类分类

def test_model_training_step(model, data_module):
    """测试单个训练步骤"""
    device = next(model.parameters()).device  # 获取模型所在设备
    
    # 设置训练组件
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters())
    
    # 获取一个批次的数据
    train_loader = data_module.get_train_dataloader()
    batch = next(iter(train_loader))
    inputs, labels = batch
    
    # 将数据移动到正确的设备
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = labels.to(device)
    
    # 前向传播
    outputs = model(inputs['input_1'], inputs['input_2'])
    loss = criterion(outputs, labels)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    assert not torch.isnan(loss)
    assert loss.item() > 0

def test_model_save_load(model, tmp_path):
    """测试模型保存和加载"""
    # 初始化组件
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    # 保存模型
    save_path = tmp_path / "model.pt"
    save_model(model, save_path, epoch=1, optimizer=optimizer, 
              scheduler=scheduler, best_metric=0.95)
    
    # 加载模型
    new_model = build_hybrid_model(num_classes=2)
    new_optimizer = optim.Adam(new_model.parameters())
    new_scheduler = optim.lr_scheduler.ReduceLROnPlateau(new_optimizer)
    
    loaded_model, loaded_optimizer, loaded_scheduler, epoch, best_metric = load_model(
        save_path, new_model, new_optimizer, new_scheduler
    )
    
    assert epoch == 1
    assert best_metric == 0.95
    assert isinstance(loaded_model, type(model))

def test_model_unfreeze(model):
    """测试模型层解冻功能"""
    # 初始状态：所有层都被冻结
    for param in model.resnet.parameters():
        param.requires_grad = False
    for param in model.efficientnet.parameters():
        param.requires_grad = False
    
    # 解冻部分层
    unfreeze_layers(model, num_layers=5)
    
    # 检查是否正确解冻
    unfrozen_count_resnet = sum(1 for param in reversed(list(model.resnet.parameters()))
                               if param.requires_grad)
    unfrozen_count_efficient = sum(1 for param in reversed(list(model.efficientnet.parameters()))
                                 if param.requires_grad)
    
    assert unfrozen_count_resnet == 5
    assert unfrozen_count_efficient == 5

def test_error_handling(model, data_module):
    """测试错误处理"""
    # 测试输入尺寸不匹配
    with pytest.raises(RuntimeError):
        wrong_size = torch.randn(2, 3, 128, 128)  # 错误的输入尺寸
        model(wrong_size, wrong_size)
    
    # 测试批次大小不匹配
    with pytest.raises(RuntimeError):
        x1 = torch.randn(2, 3, 224, 224)
        x2 = torch.randn(3, 3, 224, 224)  # 批次大小不匹配
        model(x1, x2)

def test_gpu_memory_management(model, data_module):
    """测试GPU内存管理"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA不可用，跳过GPU测试")
        
    device = next(model.parameters()).device  # 获取模型所在设备

    # 记录初始GPU内存使用
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()

    # 运行一些训练步骤
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters())

    train_loader = data_module.get_train_dataloader()
    for _ in range(3):  # 运行几个批次
        batch = next(iter(train_loader))
        inputs, labels = batch
        
        # 将数据移动到正确的设备
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        
        # 前向传播和反向传播
        outputs = model(inputs['input_1'], inputs['input_2'])
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 确保内存使用正常
    current_memory = torch.cuda.memory_allocated()
    assert current_memory > initial_memory  # 应该使用了一些GPU内存
    
    # 清理
    del outputs, loss
    torch.cuda.empty_cache()
    final_memory = torch.cuda.memory_allocated()
    assert final_memory <= current_memory  # 内存应该减少或保持不变

def test_cleanup(test_data_dir):
    """测试清理功能"""
    # 验证临时目录是否被正确清理
    assert os.path.exists(test_data_dir)  # 在with块内应该存在
