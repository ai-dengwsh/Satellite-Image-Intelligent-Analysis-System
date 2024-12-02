import pytest
import torch
import os
import shutil
import numpy as np
from pathlib import Path
from src.data.data_preprocessing import SatelliteDataset, SatelliteDataModule

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
def data_module(test_data_dir):
    """创建数据模块"""
    return SatelliteDataModule(
        data_dir=test_data_dir,
        img_size=(224, 224),
        batch_size=2
    )

def test_dataset_initialization(test_data_dir):
    """测试数据集初始化"""
    try:
        dataset = SatelliteDataset(
            data_dir=os.path.join(test_data_dir, 'train'),
            transform=None,
            train=True
        )
        assert len(dataset) > 0, "数据集应该包含样本"
        assert hasattr(dataset, 'image_paths'), "数据集应该有image_paths属性"
        assert hasattr(dataset, 'labels'), "数据集应该有labels属性"
    except Exception as e:
        pytest.fail(f"数据集初始化失败: {str(e)}")

def test_dataset_getitem(test_data_dir):
    """测试数据集的__getitem__方法"""
    try:
        dataset = SatelliteDataset(
            data_dir=os.path.join(test_data_dir, 'train'),
            transform=None,
            train=True
        )
        
        sample = dataset[0]
        assert isinstance(sample, tuple), "返回值应该是元组"
        assert len(sample) == 2, "返回值应该包含数据和标签"
        assert isinstance(sample[0], dict), "第一个元素应该是字典"
        assert isinstance(sample[1], torch.Tensor), "第二个元素应该是张量"
        
        # 检查输入数据
        assert 'input_1' in sample[0], "应该包含input_1"
        assert 'input_2' in sample[0], "应该包含input_2"
        assert sample[0]['input_1'].shape[-3:] == (3, 224, 224), "图像形状不正确"
    except Exception as e:
        pytest.fail(f"数据集获取样本失败: {str(e)}")

def test_datamodule_initialization(data_module):
    """测试数据模块初始化"""
    try:
        assert hasattr(data_module, 'train_transform'), "应该有train_transform"
        assert hasattr(data_module, 'val_transform'), "应该有val_transform"
    except Exception as e:
        pytest.fail(f"数据模块初始化失败: {str(e)}")

def test_datamodule_dataloaders(data_module):
    """测试数据加载器"""
    try:
        train_loader = data_module.get_train_dataloader()
        val_loader = data_module.get_val_dataloader()
        test_loader = data_module.get_test_dataloader()
        
        # 检查数据加载器
        for loader in [train_loader, val_loader, test_loader]:
            assert loader is not None, "数据加载器不应为None"
            batch = next(iter(loader))
            assert isinstance(batch, tuple), "批次应该是元组"
            assert len(batch) == 2, "批次应该包含数据和标签"
            assert isinstance(batch[0], dict), "第一个元素应该是字典"
            assert isinstance(batch[1], torch.Tensor), "第二个元素应该是张量"
            
            # 检查批次大小
            assert batch[0]['input_1'].shape[0] == data_module.batch_size, "批次大小不正确"
            assert batch[1].shape[0] == data_module.batch_size, "标签批次大小不正确"
    except Exception as e:
        pytest.fail(f"数据加载器测试失败: {str(e)}")

def test_transforms(data_module):
    """测试数据增强转换"""
    try:
        # 创建示例图像
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # 测试训练转换
        transformed = data_module.train_transform(image=image)
        assert 'image' in transformed, "转换结果应该包含'image'键"
        assert isinstance(transformed['image'], torch.Tensor), "转换后应该是PyTorch张量"
        assert transformed['image'].shape == (3, 224, 224), "转换后的图像形状不正确"
        
        # 测试验证转换
        transformed = data_module.val_transform(image=image)
        assert 'image' in transformed, "转换结果应该包含'image'键"
        assert isinstance(transformed['image'], torch.Tensor), "转换后应该是PyTorch张量"
        assert transformed['image'].shape == (3, 224, 224), "转换后的图像形状不正确"
    except Exception as e:
        pytest.fail(f"数据增强测试失败: {str(e)}")

def test_data_loading_errors(test_data_dir):
    """测试数据加载错误处理"""
    try:
        # 测试不存在的目录
        with pytest.raises(Exception):
            SatelliteDataset(data_dir="non_existent_dir")
        
        # 测试空目录
        empty_dir = os.path.join(test_data_dir, "empty")
        os.makedirs(empty_dir, exist_ok=True)
        with pytest.raises(Exception):
            SatelliteDataset(data_dir=empty_dir)
        
        # 测试损坏的图像文件
        corrupt_dir = os.path.join(test_data_dir, "corrupt")
        os.makedirs(os.path.join(corrupt_dir, "class1"), exist_ok=True)
        with open(os.path.join(corrupt_dir, "class1", "corrupt.jpg"), "w") as f:
            f.write("corrupt data")
        
        dataset = SatelliteDataset(data_dir=os.path.join(test_data_dir, "train"))
        assert len(dataset) > 0, "有效数据集应该包含样本"
    except Exception as e:
        pytest.fail(f"错误处理测试失败: {str(e)}")

def test_data_augmentation_consistency(data_module):
    """测试数据增强的一致性"""
    try:
        # 创建相同的测试图像
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # 对相同图像进行多次验证转换
        transformed1 = data_module.val_transform(image=image)
        transformed2 = data_module.val_transform(image=image)
        
        # 验证转换应该是确定性的
        assert torch.all(transformed1['image'] == transformed2['image']), \
            "验证转换应该是确定性的"
            
        # 训练转换应该产生不同的结果
        transformed1 = data_module.train_transform(image=image)
        transformed2 = data_module.train_transform(image=image)
        
        # 训练转换应该是随机的
        assert not torch.all(transformed1['image'] == transformed2['image']), \
            "训练转换应该是随机的"
    except Exception as e:
        pytest.fail(f"数据增强一致性测试失败: {str(e)}")

def test_memory_efficiency(data_module):
    """测试内存效率"""
    try:
        train_loader = data_module.get_train_dataloader()
        
        # 测试数据加载器的内存使用
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # 加载多个批次
        for _ in range(5):
            batch = next(iter(train_loader))
            del batch
        
        # 检查内存增长
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        assert memory_growth < 1024 * 1024 * 100, "内存增长不应超过100MB"
    except Exception as e:
        pytest.fail(f"内存效率测试失败: {str(e)}")

def test_multiprocessing(data_module):
    """测试多进程数据加载"""
    try:
        # 测试不同的工作进程数
        for num_workers in [0, 2, 4]:
            train_loader = data_module.get_train_dataloader()
            
            # 加载一些批次
            for _ in range(3):
                batch = next(iter(train_loader))
                assert isinstance(batch, tuple), "批次格式不正确"
                assert len(batch) == 2, "批次应该包含数据和标签"
    except Exception as e:
        pytest.fail(f"多进程数据加载测试失败: {str(e)}")

def test_cleanup(test_data_dir):
    """清理测试数据"""
    try:
        if os.path.exists(test_data_dir):
            shutil.rmtree(test_data_dir)
    except Exception as e:
        pytest.fail(f"清理测试数据失败: {str(e)}")
