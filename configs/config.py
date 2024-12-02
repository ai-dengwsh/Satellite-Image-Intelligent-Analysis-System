"""
配置文件
"""

import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# 数据配置
IMAGE_SIZE = (224, 224)  # 调整为标准的ResNet/EfficientNet输入大小
BATCH_SIZE = 16  # 减小批量大小以适应显存
NUM_CLASSES = 10  # EuroSAT的10个类别

# 模型配置
INITIAL_LEARNING_RATE = 1e-4
FINE_TUNING_LEARNING_RATE = 1e-5
EPOCHS = 100  # 增加训练轮数以获得更好的性能
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.2
MIN_LR = 1e-6

# 混合模型配置
TRANSFORMER_NUM_HEADS = 8
TRANSFORMER_KEY_DIM = 128
DROPOUT_RATE = 0.5
DENSE_UNITS = [512, 256]

# 异常检测配置
ANOMALY_THRESHOLD = 0.05
RECONSTRUCTION_THRESHOLD = 0.1

# 训练配置
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# 数据增强配置
ROTATION_RANGE = 30
SHIFT_RANGE = 0.2
SHEAR_RANGE = 0.2
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True
FILL_MODE = 'nearest'

# 可视化配置
VISUALIZATION_COLS = 5
VISUALIZATION_FIGSIZE = (15, 15)
ANOMALY_OVERLAY_ALPHA = 0.5

# 类别映射
CLASS_NAMES = [
    'AnnualCrop',
    'Forest',
    'HerbaceousVegetation',
    'Highway',
    'Industrial',
    'Pasture',
    'PermanentCrop',
    'Residential',
    'River',
    'SeaLake'
]
