import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model, data_loader, model_dir, model_name):
        """
        初始化训练器
        
        Args:
            model: Keras模型
            data_loader: 数据加载器实例
            model_dir: 模型保存目录
            model_name: 模型名称
        """
        self.model = model
        self.data_loader = data_loader
        self.model_dir = model_dir
        self.model_name = model_name
        
        # 创建模型保存目录
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def get_callbacks(self):
        """获取回调函数"""
        callbacks = [
            ModelCheckpoint(
                os.path.join(self.model_dir, f'{self.model_name}_best.keras'),
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6
            )
        ]
        return callbacks
    
    def train(self, epochs=50, initial_epoch=0):
        """
        训练模型
        
        Args:
            epochs (int): 训练轮数
            initial_epoch (int): 初始轮数
        """
        # 获取数据生成器
        train_generator = self.data_loader.get_train_generator()
        val_generator = self.data_loader.get_val_generator()
        
        # 训练模型
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=self.get_callbacks()
        )
        
        return history
    
    def evaluate(self):
        """评估模型"""
        test_generator = self.data_loader.get_test_generator()
        results = self.model.evaluate(test_generator)
        metrics = dict(zip(self.model.metrics_names, results))
        print("\nTest Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        return metrics
    
    @staticmethod
    def plot_training_history(history):
        """
        绘制训练历史
        
        Args:
            history: 训练历史对象
        """
        plt.figure(figsize=(12, 4))
        
        # 绘制准确率
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # 绘制损失
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_indices):
        """
        绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            class_indices: 类别索引字典
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_indices.keys(),
                   yticklabels=class_indices.keys())
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
