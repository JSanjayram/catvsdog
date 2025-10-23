import tensorflow as tf
import subprocess
import sys

def check_system():
    print("🔍 System Check:")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Python version: {sys.version}")
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\nGPU Status: {len(gpus)} GPU(s) found")
    
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
    else:
        print("❌ No GPU detected")
        print("\n💡 Solutions:")
        print("1. Install NVIDIA GPU drivers")
        print("2. Install CUDA toolkit")
        print("3. Install tensorflow-gpu:")
        print("   pip uninstall tensorflow")
        print("   pip install tensorflow[and-cuda]")
        print("\n⚡ CPU training will work but be slower")
    
    # Test computation
    try:
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print(f"\n✅ TensorFlow working on {'GPU' if gpus else 'CPU'}")
    except Exception as e:
        print(f"\n❌ TensorFlow error: {e}")

if __name__ == "__main__":
    check_system()