import tensorflow as tf

def check_gpu():
    """Check GPU availability and configuration"""
    print("🔍 GPU Check:")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check for GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ {len(gpus)} GPU(s) found:")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
        
        # Check CUDA
        print(f"CUDA available: {tf.test.is_built_with_cuda()}")
        print(f"GPU available: {tf.test.is_gpu_available()}")
        
        # Memory info
        try:
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            print(f"GPU details: {gpu_details}")
        except:
            pass
            
    else:
        print("❌ No GPU found")
        print("Training will use CPU (slower)")
    
    print("\n🚀 Speed comparison:")
    print("With GPU: ~2-5 minutes training")
    print("With CPU: ~10-20 minutes training")

if __name__ == "__main__":
    check_gpu()