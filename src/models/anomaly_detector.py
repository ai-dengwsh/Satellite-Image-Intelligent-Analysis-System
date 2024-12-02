import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Reshape, Flatten, LeakyReLU
from tensorflow.keras.models import Model
import numpy as np

class AnomalyDetector:
    """卫星图像异常检测器"""
    
    def __init__(self, input_shape=(224, 224, 3)):
        """
        初始化异常检测器
        
        Args:
            input_shape (tuple): 输入图像形状
        """
        self.input_shape = input_shape
        
    def build_autoencoder(self):
        """构建自编码器模型"""
        input_img = Input(shape=self.input_shape)
        
        # 编码器
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        
        # 解码器
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        
        autoencoder = Model(input_img, decoded)
        return autoencoder
    
    def build_gan(self):
        """构建GAN模型"""
        def build_generator():
            model = tf.keras.Sequential([
                Dense(7*7*256, use_bias=False, input_shape=(100,)),
                LeakyReLU(),
                Reshape((7, 7, 256)),
                
                Conv2D(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
                LeakyReLU(),
                
                Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
                LeakyReLU(),
                
                Conv2D(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
            ])
            return model
            
        def build_discriminator():
            model = tf.keras.Sequential([
                Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=self.input_shape),
                LeakyReLU(),
                
                Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
                LeakyReLU(),
                
                Flatten(),
                Dense(1)
            ])
            return model
            
        # 构建和编译判别器
        discriminator = build_discriminator()
        discriminator.compile(optimizer='adam',
                            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                            metrics=['accuracy'])
        
        # 构建生成器
        generator = build_generator()
        
        # 构建GAN
        discriminator.trainable = False
        gan_input = Input(shape=(100,))
        x = generator(gan_input)
        gan_output = discriminator(x)
        gan = Model(gan_input, gan_output)
        gan.compile(optimizer='adam',
                   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
        
        return generator, discriminator, gan
    
    @staticmethod
    def detect_anomalies(model, images, threshold=0.05):
        """
        使用训练好的模型检测异常
        
        Args:
            model: 训练好的模型
            images: 输入图像
            threshold: 异常阈值
            
        Returns:
            anomalies: 异常检测结果
            reconstruction_error: 重建误差
        """
        reconstructed = model.predict(images)
        reconstruction_error = np.mean(np.square(images - reconstructed), axis=(1,2,3))
        anomalies = reconstruction_error > threshold
        return anomalies, reconstruction_error
