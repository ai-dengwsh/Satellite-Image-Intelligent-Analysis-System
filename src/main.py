import os
import sys
import argparse
from configs.config import *
from src.data.data_preprocessing import SatelliteDataGenerator
from src.models.hybrid_model import build_hybrid_model, unfreeze_model
from src.models.anomaly_detector import AnomalyDetector
from src.training.train import ModelTrainer
from src.utils.visualization import Visualizer

def ensure_directories():
    """确保所有必要的目录都存在"""
    directories = [DATA_DIR, MODEL_DIR, os.path.join(MODEL_DIR, 'classifier'), os.path.join(MODEL_DIR, 'anomaly')]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def train_classifier(args):
    """训练分类器"""
    # 确保目录存在
    ensure_directories()
    
    # 初始化数据加载器
    data_generator = SatelliteDataGenerator(
        data_dir=DATA_DIR,
        img_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # 构建混合模型
    model = build_hybrid_model(input_shape=IMAGE_SIZE + (3,), num_classes=NUM_CLASSES)
    
    # 初始化训练器
    trainer = ModelTrainer(
        model=model,
        data_loader=data_generator,
        model_dir=os.path.join(MODEL_DIR, 'classifier'),
        model_name='satellite_classifier_hybrid'
    )
    
    # 第一阶段训练（冻结基础模型）
    print("Stage 1: Training with frozen base models...")
    history1 = trainer.train(epochs=EPOCHS // 2)
    
    # 解冻部分层进行微调
    print("\nStage 2: Fine-tuning with unfrozen layers...")
    model = unfreeze_model(model, num_layers=10)
    history2 = trainer.train(epochs=EPOCHS, initial_epoch=EPOCHS // 2)
    
    # 评估模型
    trainer.evaluate()
    
    # 可视化训练过程
    Visualizer.plot_training_metrics({
        'Accuracy': history1.history['accuracy'] + history2.history['accuracy'],
        'Loss': history1.history['loss'] + history2.history['loss'],
        'Val Accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
        'Val Loss': history1.history['val_loss'] + history2.history['val_loss']
    })

def train_anomaly_detector(args):
    """训练异常检测器"""
    # 确保目录存在
    ensure_directories()
    
    # 初始化数据加载器
    data_generator = SatelliteDataGenerator(
        data_dir=DATA_DIR,
        img_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # 初始化异常检测器
    detector = AnomalyDetector(input_shape=IMAGE_SIZE + (3,))
    
    if args.model_type == 'autoencoder':
        model = detector.build_autoencoder()
    else:  # gan
        generator, discriminator, gan = detector.build_gan()
        model = gan  # 使用GAN进行训练
    
    # 初始化训练器
    trainer = ModelTrainer(
        model=model,
        data_loader=data_generator,
        model_dir=os.path.join(MODEL_DIR, 'anomaly'),
        model_name=f'anomaly_detector_{args.model_type}'
    )
    
    # 训练模型
    history = trainer.train(epochs=EPOCHS)
    
    # 可视化训练过程
    Visualizer.plot_training_metrics({
        'Loss': history.history['loss'],
        'Val Loss': history.history['val_loss']
    })

def main():
    parser = argparse.ArgumentParser(description='卫星图像智能分析系统')
    parser.add_argument('--task', type=str, required=True,
                      choices=['classification', 'anomaly_detection'],
                      help='训练任务类型')
    parser.add_argument('--model_type', type=str, required=True,
                      choices=['hybrid', 'autoencoder', 'gan'],
                      help='模型类型')
    
    args = parser.parse_args()
    
    if args.task == 'classification':
        train_classifier(args)
    else:  # anomaly_detection
        train_anomaly_detector(args)

if __name__ == '__main__':
    main()
