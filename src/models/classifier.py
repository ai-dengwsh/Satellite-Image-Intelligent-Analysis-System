import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, Concatenate, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model

class SatelliteImageClassifier:
    """卫星图像分类器"""
    
    def __init__(self, num_classes, input_shape=(224, 224, 3)):
        """
        初始化分类器
        
        Args:
            num_classes (int): 类别数量
            input_shape (tuple): 输入图像形状
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        
    def build_resnet_model(self):
        """构建基于ResNet50的模型"""
        # 使用更小的ResNet18或自定义的轻量级模型
        def build_light_resnet():
            inputs = Input(shape=self.input_shape)
            x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
            x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D((2, 2))(x)
            
            x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D((2, 2))(x)
            
            x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D((2, 2))(x)
            
            x = GlobalAveragePooling2D()(x)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.5)(x)
            outputs = Dense(self.num_classes, activation='softmax')(x)
            
            return Model(inputs, outputs)
        
        model = build_light_resnet()
        return model
    
    def build_efficientnet_model(self):
        """构建基于EfficientNetB0的模型"""
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=self.input_shape)
        
        # 冻结预训练层
        base_model.trainable = False
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
    
    def build_ensemble_model(self):
        """构建集成模型"""
        # ResNet50分支
        resnet = self.build_resnet_model()
        
        # EfficientNetB0分支
        efficientnet = EfficientNetB0(weights='imagenet', include_top=False, input_shape=self.input_shape)
        efficientnet.trainable = False
        
        # 定义输入
        input_tensor = Input(shape=self.input_shape)
        
        # ResNet分支
        x1 = resnet(input_tensor)
        
        # EfficientNet分支
        x2 = efficientnet(input_tensor)
        x2 = GlobalAveragePooling2D()(x2)
        x2 = Dense(512, activation='relu')(x2)
        x2 = Dropout(0.5)(x2)
        
        # 合并分支
        combined = Concatenate()([x1, x2])
        combined = Dense(256, activation='relu')(combined)
        combined = Dropout(0.5)(combined)
        predictions = Dense(self.num_classes, activation='softmax')(combined)
        
        model = Model(inputs=input_tensor, outputs=predictions)
        return model
    
    @staticmethod
    def compile_model(model, learning_rate=1e-4):
        """
        编译模型
        
        Args:
            model: Keras模型
            learning_rate (float): 学习率
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
