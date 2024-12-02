import pytest
import torch
import os
import shutil
import numpy as np
from pathlib import Path
from src.models.hybrid_model import HybridModel
from src.data.data_preprocessing import SatelliteDataModule

@pytest.fixture
def test_data_dir(tmp_path):
    """创建测试数据目录"""
    data_dir = tmp_path / "test_data"
    splits = ['train', 'val', 'test']
    classes = ['class1', 'class2']
    
    # 创建目录结构
    for split in splits:
        for cls in classes:
            (data_dir / split / cls).mkdir(parents=True)
            
            # 创建测试图像
            for i in range(5):
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img_path = data_dir / split / cls / f"img_{i}.jpg"
                import cv2
                cv2.imwrite(str(img_path), img)
    
    return str(data_dir)

@pytest.fixture
def model():
    """创建模型"""
    return HybridModel(num_classes=2)

@pytest.fixture
def data_module(test_data_dir):
    """创建数据模块"""
    return SatelliteDataModule(
        data_dir=test_data_dir,
        img_size=(224, 224),
        batch_size=2
    )

def test_full_training_cycle(model, data_module):
    """测试完整的训练周期"""
    try:
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # 准备数据加载器
        train_loader = data_module.get_train_dataloader()
        val_loader = data_module.get_val_dataloader()
        
        # 设置训练参数
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # 训练一个epoch
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 2:  # 只训练几个批次用于测试
                break
                
            # 将数据移到设备上
            input_1 = data['input_1'].to(device)
            input_2 = data['input_2'].to(device)
            target = target.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(input_1, input_2)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 验证损失值
            assert not torch.isnan(loss), "损失值不应为NaN"
            assert not torch.isinf(loss), "损失值不应为Inf"
        
        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                if batch_idx >= 2:  # 只验证几个批次
                    break
                    
                input_1 = data['input_1'].to(device)
                input_2 = data['input_2'].to(device)
                target = target.to(device)
                
                output = model(input_1, input_2)
                val_loss += criterion(output, target).item()
                
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        # 验证结果
        assert val_loss > 0, "验证损失应该大于0"
        assert correct >= 0, "正确预测数不应为负"
    except Exception as e:
        pytest.fail(f"训练周期测试失败: {str(e)}")

def test_model_save_load(model, data_module, tmp_path):
    """测试模型保存和加载"""
    try:
        # 准备测试数据
        train_loader = data_module.get_train_dataloader()
        batch = next(iter(train_loader))
        
        # 获取初始预测
        model.eval()
        with torch.no_grad():
            initial_output = model(batch[0]['input_1'], batch[0]['input_2'])
        
        # 保存模型
        save_path = tmp_path / "test_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
        }, save_path)
        
        # 加载模型到新实例
        new_model = HybridModel(num_classes=2)
        checkpoint = torch.load(save_path)
        new_model.load_state_dict(checkpoint['model_state_dict'])
        
        # 比较预测结果
        new_model.eval()
        with torch.no_grad():
            loaded_output = new_model(batch[0]['input_1'], batch[0]['input_2'])
        
        assert torch.allclose(initial_output, loaded_output), "加载后的模型预测结果应该相同"
    except Exception as e:
        pytest.fail(f"模型保存加载测试失败: {str(e)}")

def test_inference_pipeline(model, data_module):
    """测试推理流程"""
    try:
        # 准备测试数据
        test_loader = data_module.get_test_dataloader()
        
        # 设置为评估模式
        model.eval()
        
        # 进行推理
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(test_loader):
                if batch_idx >= 2:  # 只测试几个批次
                    break
                    
                # 前向传播
                output = model(data['input_1'], data['input_2'])
                
                # 获取预测和概率
                probabilities = torch.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # 验证结果
        assert len(all_predictions) > 0, "应该有预测结果"
        assert all(0 <= p <= 1 for probs in all_probabilities for p in probs), \
            "概率值应该在[0,1]范围内"
    except Exception as e:
        pytest.fail(f"推理流程测试失败: {str(e)}")

def test_error_handling(model, data_module):
    """测试错误处理"""
    try:
        # 测试错误的输入尺寸
        with pytest.raises(Exception):
            wrong_size_input = torch.randn(1, 3, 32, 32)  # 错误的输入尺寸
            model(wrong_size_input, wrong_size_input)
        
        # 测试不匹配的输入
        with pytest.raises(Exception):
            input1 = torch.randn(1, 3, 224, 224)
            input2 = torch.randn(2, 3, 224, 224)  # 批次大小不匹配
            model(input1, input2)
        
        # 测试无效的状态字典
        with pytest.raises(Exception):
            model.load_state_dict({"invalid": "state_dict"})
    except Exception as e:
        pytest.fail(f"错误处理测试失败: {str(e)}")

def test_gpu_memory_management(model, data_module):
    """测试GPU内存管理"""
    if not torch.cuda.is_available():
        pytest.skip("GPU不可用，跳过测试")
    
    try:
        # 移动模型到GPU
        model = model.cuda()
        
        # 记录初始GPU内存
        initial_memory = torch.cuda.memory_allocated()
        
        # 运行一些操作
        train_loader = data_module.get_train_dataloader()
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 2:  # 只测试几个批次
                break
            
            input_1 = data['input_1'].cuda()
            input_2 = data['input_2'].cuda()
            output = model(input_1, input_2)
            del output, input_1, input_2
        
        # 强制GPU内存回收
        torch.cuda.empty_cache()
        
        # 检查内存是否正确释放
        final_memory = torch.cuda.memory_allocated()
        memory_diff = abs(final_memory - initial_memory)
        assert memory_diff < 1024 * 1024 * 10, "GPU内存泄漏检测"
    except Exception as e:
        pytest.fail(f"GPU内存管理测试失败: {str(e)}")

def test_cleanup(test_data_dir):
    """清理测试数据"""
    try:
        if os.path.exists(test_data_dir):
            shutil.rmtree(test_data_dir)
    except Exception as e:
        pytest.fail(f"清理测试数据失败: {str(e)}")
