# Model Configuration
MODEL_PATH = 'cat_dog_model.h5'
IMG_SIZE = (128, 128)
BATCH_SIZE = 128  # GPU optimized
EPOCHS = 10
LEARNING_RATE = 0.001

# Class Labels
CLASSES = {
    0: 'cat',
    1: 'dog', 
    2: 'other'
}

CLASS_NAMES = ['cat', 'dog', 'other']

# Confidence Thresholds
HIGH_CONFIDENCE = 0.90
MEDIUM_CONFIDENCE = 0.70
LOW_CONFIDENCE = 0.50

# Data Augmentation Parameters
ROTATION_RANGE = 40
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
SHEAR_RANGE = 0.2
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True

# Training Parameters
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 3
REDUCE_LR_PATIENCE = 2
REDUCE_LR_FACTOR = 0.5

# Dataset URLs for massive training data
DATASET_URLS = {
    'cats_dogs': 'https://www.microsoft.com/en-us/download/details.aspx?id=54765',
    'oxford_pets': 'https://www.robots.ox.ac.uk/~vgg/data/pets/',
    'stanford_dogs': 'http://vision.stanford.edu/aditya86/ImageNetDogs/',
    'cats_vs_dogs_redux': 'https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition'
}

# Model Architecture
BASE_MODEL = 'MobileNetV2'
FINE_TUNE_LAYERS = 20
DROPOUT_RATE = 0.2
DENSE_UNITS = 64