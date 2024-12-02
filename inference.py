import torch
import cv2
import numpy as np
from src.models.hybrid_model import HybridModel
import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_model(model_path, num_classes=2, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    model = HybridModel(num_classes=num_classes)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model

def preprocess_image(image_path, img_size=(224, 224)):
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 定义预处理转换
    transform = A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # 应用转换
    transformed = transform(image=image)
    return transformed["image"]

def predict(model, image_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 预处理图像
    image = preprocess_image(image_path)
    image = image.unsqueeze(0)  # 添加批次维度
    
    # 将图像移到设备上
    image = image.to(device)
    
    # 进行预测
    with torch.no_grad():
        outputs = model(image, image)  # 使用相同的图像作为两个输入
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence

if __name__ == '__main__':
    # 示例用法
    model_path = 'best_model.pth'
    image_path = 'path/to/your/image.jpg'  # 替换为您的图像路径
    
    # 加载模型
    model = load_model(model_path)
    
    # 进行预测
    predicted_class, confidence = predict(model, image_path)
    print(f'Predicted class: {predicted_class}')
    print(f'Confidence: {confidence:.2%}')
