import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import os

# GPU Configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print(f"🚀 GPU acceleration enabled: {len(gpus)} GPU(s) found")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("⚠️ No GPU found, using CPU (still fast!)")
    print("💡 To enable GPU: pip install tensorflow[and-cuda]")

class CatDogClassifier:
    def __init__(self):
        self.model = None
        self.img_size = (128, 128)
        self.classes = ['cat', 'dog', 'other']
        
    def create_model(self):
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(128, 128, 3)
        )
        base_model.trainable = False
        
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(3, activation='softmax', dtype='float32')
        ])
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def train_model(self, train_dir, val_dir, epochs=20):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            horizontal_flip=True
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Optimized batch size
        batch_size = 128 if gpus else 32
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
        ]
        
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks
        )
        
        return history
    
    def predict(self, image):
        if isinstance(image, str):
            image = Image.open(image)
        
        image = image.convert('RGB')
        image = image.resize(self.img_size)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        predictions = self.model.predict(image_array)
        confidence = float(np.max(predictions))
        predicted_class = self.classes[np.argmax(predictions)]
        
        return predicted_class, confidence
    
    def save_model(self, path='cat_dog_model.h5'):
        self.model.save(path)
    
    def load_model(self, path='cat_dog_model.h5'):
        self.model = tf.keras.models.load_model(path)