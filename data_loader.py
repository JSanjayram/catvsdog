import os
import requests
import zipfile
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from constants import *

class DataLoader:
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_dataset(self, url, extract_path):
        """Download and extract dataset"""
        filename = url.split('/')[-1]
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        if filename.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
    
    def create_data_generators(self, train_dir, val_dir):
        """Create training and validation data generators"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=ROTATION_RANGE,
            width_shift_range=WIDTH_SHIFT_RANGE,
            height_shift_range=HEIGHT_SHIFT_RANGE,
            shear_range=SHEAR_RANGE,
            zoom_range=ZOOM_RANGE,
            horizontal_flip=HORIZONTAL_FLIP,
            brightness_range=[0.8, 1.2],
            channel_shift_range=0.1,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def setup_directories(self):
        """Setup training directory structure"""
        dirs = ['train/cat', 'train/dog', 'train/other', 
                'val/cat', 'val/dog', 'val/other']
        
        for dir_path in dirs:
            (self.data_dir / dir_path).mkdir(parents=True, exist_ok=True)
        
        return {
            'train': self.data_dir / 'train',
            'val': self.data_dir / 'val'
        }
    
    def get_dataset_info(self, directory):
        """Get information about dataset"""
        total_images = 0
        class_counts = {}
        
        for class_dir in directory.iterdir():
            if class_dir.is_dir():
                count = len(list(class_dir.glob('*')))
                class_counts[class_dir.name] = count
                total_images += count
        
        return total_images, class_counts