import tensorflow as tf
from model import CatDogClassifier
from data_loader import DataLoader
from constants import *
import matplotlib.pyplot as plt

def train_high_accuracy_model():
    """Train model with 90%+ accuracy using massive dataset"""
    
    # Initialize components
    classifier = CatDogClassifier()
    data_loader = DataLoader()
    
    # Setup data directories
    dirs = data_loader.setup_directories()
    
    print("Setting up training environment...")
    
    # Create model
    classifier.create_model()
    
    # Print model summary
    classifier.model.summary()
    
    # Create data generators with massive augmentation
    train_gen, val_gen = data_loader.create_data_generators(
        dirs['train'], dirs['val']
    )
    
    # Enhanced callbacks for high accuracy
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            min_delta=0.001
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    print("Starting training for 90%+ accuracy...")
    
    # Train model
    history = classifier.model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Fine-tuning for higher accuracy
    print("Fine-tuning for maximum accuracy...")
    
    # Unfreeze top layers
    classifier.model.layers[0].trainable = True
    for layer in classifier.model.layers[0].layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    classifier.model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE/10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training
    history_fine = classifier.model.fit(
        train_gen,
        epochs=5,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    classifier.save_model(MODEL_PATH)
    
    # Evaluate final accuracy
    val_loss, val_accuracy = classifier.model.evaluate(val_gen)
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")
    
    if val_accuracy >= 0.90:
        print("✅ Target accuracy of 90%+ achieved!")
    else:
        print("⚠️ Target accuracy not reached. Consider more training data or epochs.")
    
    return classifier, history

def plot_training_history(history):
    """Plot training metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

if __name__ == "__main__":
    classifier, history = train_high_accuracy_model()
    plot_training_history(history)