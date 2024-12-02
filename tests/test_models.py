import pytest
import torch
import torch.nn as nn
from src.models.hybrid_model import HybridModel

@pytest.fixture
def model():
    return HybridModel(num_classes=2)

@pytest.fixture
def sample_batch():
    batch_size = 4
    channels = 3
    height = 224
    width = 224
    return {
        'input_1': torch.randn(batch_size, channels, height, width),
        'input_2': torch.randn(batch_size, channels, height, width)
    }

def test_model_initialization(model):
    """测试模型初始化"""
    assert isinstance(model, nn.Module), "模型应该是nn.Module的子类"
    assert isinstance(model.resnet, nn.Module), "ResNet应该是nn.Module的子类"
    assert isinstance(model.efficientnet, nn.Module), "EfficientNet应该是nn.Module的子类"

def test_model_forward(model, sample_batch):
    """测试模型前向传播"""
    try:
        model.eval()
        with torch.no_grad():
            output = model(sample_batch['input_1'], sample_batch['input_2'])
        
        assert isinstance(output, torch.Tensor), "输出应该是torch.Tensor类型"
        assert output.shape[0] == sample_batch['input_1'].shape[0], "批次大小应该保持一致"
        assert output.shape[1] == 2, "输出维度应该等于类别数"
    except Exception as e:
        pytest.fail(f"前向传播失败: {str(e)}")

def test_model_device_compatibility(model, sample_batch):
    """测试模型在不同设备上的兼容性"""
    if torch.cuda.is_available():
        try:
            model = model.cuda()
            cuda_input1 = sample_batch['input_1'].cuda()
            cuda_input2 = sample_batch['input_2'].cuda()
            
            with torch.no_grad():
                output = model(cuda_input1, cuda_input2)
            
            assert output.is_cuda, "输出应该在GPU上"
        except Exception as e:
            pytest.fail(f"GPU兼容性测试失败: {str(e)}")

def test_model_gradient_flow(model, sample_batch):
    """测试模型梯度流"""
    try:
        model.train()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        # 创建假标签
        labels = torch.randint(0, 2, (sample_batch['input_1'].shape[0],))
        
        # 前向传播
        outputs = model(sample_batch['input_1'], sample_batch['input_2'])
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 检查梯度
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} 的梯度为None"
                assert not torch.isnan(param.grad).any(), f"{name} 的梯度包含NaN"
                assert not torch.isinf(param.grad).any(), f"{name} 的梯度包含Inf"
    except Exception as e:
        pytest.fail(f"梯度流测试失败: {str(e)}")

def test_model_input_size_handling(model):
    """测试模型对不同输入尺寸的处理"""
    test_sizes = [(3, 224, 224), (3, 256, 256), (3, 299, 299)]
    
    for size in test_sizes:
        try:
            batch = {
                'input_1': torch.randn(1, *size),
                'input_2': torch.randn(1, *size)
            }
            with torch.no_grad():
                output = model(batch['input_1'], batch['input_2'])
            assert output.shape[1] == 2, f"输入尺寸 {size} 的输出类别数不正确"
        except Exception as e:
            pytest.fail(f"输入尺寸 {size} 测试失败: {str(e)}")

def test_model_batch_size_handling(model):
    """测试模型对不同批次大小的处理"""
    test_batch_sizes = [1, 4, 8, 16]
    
    for batch_size in test_batch_sizes:
        try:
            batch = {
                'input_1': torch.randn(batch_size, 3, 224, 224),
                'input_2': torch.randn(batch_size, 3, 224, 224)
            }
            with torch.no_grad():
                output = model(batch['input_1'], batch['input_2'])
            assert output.shape[0] == batch_size, f"批次大小 {batch_size} 的输出形状不正确"
        except Exception as e:
            pytest.fail(f"批次大小 {batch_size} 测试失败: {str(e)}")

def test_model_state_dict(model):
    """测试模型状态字典的保存和加载"""
    try:
        # 保存状态字典
        state_dict = model.state_dict()
        
        # 创建新模型并加载状态字典
        new_model = HybridModel(num_classes=2)
        new_model.load_state_dict(state_dict)
        
        # 比较两个模型的参数
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert torch.equal(param1, param2), f"参数 {name1} 在加载后不匹配"
    except Exception as e:
        pytest.fail(f"状态字典测试失败: {str(e)}")

def test_model_freeze_unfreeze(model):
    """测试模型参数的冻结和解冻"""
    try:
        # 测试初始状态（预训练层应该被冻结）
        for param in model.resnet.parameters():
            assert not param.requires_grad, "ResNet预训练层应该被冻结"
        for param in model.efficientnet.parameters():
            assert not param.requires_grad, "EfficientNet预训练层应该被冻结"
        
        # 测试解冻功能
        for param in model.parameters():
            param.requires_grad = True
        
        # 验证解冻状态
        for param in model.parameters():
            assert param.requires_grad, "参数应该被解冻"
    except Exception as e:
        pytest.fail(f"参数冻结测试失败: {str(e)}")
