import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from src.models.hybrid_model import HybridModel
from src.data.data_preprocessing import SatelliteDataModule
import os
from tqdm import tqdm

def train_model(
    data_dir,
    num_epochs=100,
    batch_size=32,
    learning_rate=1e-4,
    num_classes=2,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # 初始化数据模块
    data_module = SatelliteDataModule(data_dir, batch_size=batch_size)
    train_loader = data_module.get_train_dataloader()
    val_loader = data_module.get_val_dataloader()
    
    # 初始化模型
    model = HybridModel(num_classes=num_classes)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_data, labels in train_bar:
            # 将数据移到设备上
            input_1 = batch_data['input_1'].to(device)
            input_2 = batch_data['input_2'].to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(input_1, input_2)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            train_bar.set_postfix({
                'loss': f'{train_loss/train_total:.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch_data, labels in val_bar:
                input_1 = batch_data['input_1'].to(device)
                input_2 = batch_data['input_2'].to(device)
                labels = labels.to(device)
                
                outputs = model(input_1, input_2)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_bar.set_postfix({
                    'loss': f'{val_loss/val_total:.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        # 计算平均损失和准确率
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100. * train_correct / train_total
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100. * val_correct / val_total
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
        print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
        
        # 学习率调整
        scheduler.step(epoch_val_loss)
        
        # 保存最佳模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, 'best_model.pth')
            print(f'Saved best model with validation loss: {best_val_loss:.4f}')

if __name__ == '__main__':
    data_dir = 'data'  # 替换为您的数据目录路径
    train_model(
        data_dir=data_dir,
        num_epochs=100,
        batch_size=32,
        learning_rate=1e-4,
        num_classes=2  # 根据您的类别数调整
    )
