# 卫星图像智能分析系统

基于深度学习的卫星图像智能分类与异常检测系统。

## 项目结构

```
satellite_image_analysis/
├── data/                   # 数据目录
│   ├── raw/               # 原始数据
│   ├── processed/         # 处理后的数据
│   ├── train/            # 训练集
│   ├── val/              # 验证集
│   └── test/             # 测试集
├── models/                # 模型目录
│   ├── classifier/       # 分类模型
│   └── anomaly/         # 异常检测模型
├── src/                   # 源代码
│   ├── data/            # 数据处理模块
│   ├── models/          # 模型定义模块
│   ├── training/        # 训练相关模块
│   └── utils/           # 工具函数
├── notebooks/            # Jupyter notebooks
├── tests/                # 测试代码
├── configs/              # 配置文件
└── requirements.txt      # 项目依赖
```

## 环境要求

- Python 3.8+
- CUDA支持的NVIDIA GPU（推荐）
- 16GB+ RAM
- 500GB+ 存储空间

## 安装

1. 克隆仓库：
```bash
git clone [repository-url]
cd satellite_image_analysis
```

2. 创建虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用说明

1. 数据准备：
```bash
python src/data/prepare_data.py
```

2. 训练模型：
```bash
python src/training/train_classifier.py
python src/training/train_anomaly_detector.py
```

3. 预测：
```bash
python src/models/predict.py --input [image_path]
```

## 主要功能

1. 卫星图像分类
   - 支持多类别分类
   - 基于ResNet50和EfficientNet的迁移学习
   - 集成Vision Transformers提升性能

2. 异常检测
   - 基于自编码器的异常检测
   - GAN模型支持
   - 实时异常区域标注

3. 结果可视化
   - 训练过程可视化
   - 预测结果展示
   - 异常区域标注

## 技术特点

- 多模型集成
- 自监督学习
- 知识蒸馏
- 模型优化（剪枝、量化）
- 高效数据处理

## 贡献指南

欢迎提交问题和改进建议！

## 许可证

MIT License
