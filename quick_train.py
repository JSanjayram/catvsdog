import tensorflow as tf
from model import CatDogClassifier
from data_loader import DataLoader
from constants import *

def quick_train():
    """Ultra-fast training for immediate results"""
    
    print("Starting FAST training...")
    
    # Initialize
    classifier = CatDogClassifier()
    data_loader = DataLoader()
    dirs = data_loader.setup_directories()
    
    # Create model
    classifier.create_model()
    
    # Fast data generators
    train_gen, val_gen = data_loader.create_data_generators(
        dirs['train'], dirs['val']
    )
    
    # Minimal callbacks for speed
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
    ]
    
    # Train fast
    history = classifier.model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model
    classifier.save_model(MODEL_PATH)
    
    # Quick evaluation
    val_loss, val_accuracy = classifier.model.evaluate(val_gen, verbose=0)
    print(f"Final Accuracy: {val_accuracy:.3f}")
    
    return classifier

if __name__ == "__main__":
    quick_train()