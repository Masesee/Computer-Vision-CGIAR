from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D
from tensorflow.keras.layers import Concatenate, UpSampling2D, Reshape

def build_yolo(input_shape, num_classes=80, anchors_per_scale=3):
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Darknet-like backbone
    # Initial convolution
    x = Conv2D(32, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Downsampling blocks
    x = _downsample_block(x, 64)
    x = _downsample_block(x, 128)
    x = route1 = _downsample_block(x, 256)
    x = route2 = _downsample_block(x, 512)
    x = _downsample_block(x, 1024)
    
    # Neck (FPN-like structure)
    x = _conv_block(x, 512, 1)
    x = _conv_block(x, 1024, 3)
    x = _conv_block(x, 512, 1)
    
    # First detection branch (large objects)
    large_branch = _conv_block(x, 1024, 3)
    large_output = _detection_layer(large_branch, anchors_per_scale, num_classes)
    
    # Upsample and concat with route2
    x = _conv_block(x, 256, 1)
    x = UpSampling2D()(x)
    x = Concatenate()([x, route2])
    
    # Middle detection branch (medium objects)
    x = _conv_block(x, 256, 1)
    x = _conv_block(x, 512, 3)
    x = _conv_block(x, 256, 1)
    medium_branch = _conv_block(x, 512, 3)
    medium_output = _detection_layer(medium_branch, anchors_per_scale, num_classes)
    
    # Upsample and concat with route1
    x = _conv_block(x, 128, 1)
    x = UpSampling2D()(x)
    x = Concatenate()([x, route1])
    
    # Final detection branch (small objects)
    x = _conv_block(x, 128, 1)
    x = _conv_block(x, 256, 3)
    x = _conv_block(x, 128, 1)
    small_branch = _conv_block(x, 256, 3)
    small_output = _detection_layer(small_branch, anchors_per_scale, num_classes)
    
    # Create model with multiple outputs for different scales
    model = Model(inputs, [small_output, medium_output, large_output])
    
    return model

def _conv_block(x, filters, kernel_size):
    """Basic convolution block with batch normalization and leaky ReLU"""
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def _downsample_block(x, filters):
    """Downsampling block with additional convolutions"""
    x = Conv2D(filters, 3, strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Additional convolutions
    x = _conv_block(x, filters//2, 1)
    x = _conv_block(x, filters, 3)
    return x

def _detection_layer(x, anchors_per_scale, num_classes):
    """Detection layer for YOLO"""
    # num_outputs = anchors_per_scale * (5 + num_classes)
    # 5 = 4 box coordinates + 1 objectness score
    num_outputs = anchors_per_scale * (5 + num_classes)
    
    x = Conv2D(num_outputs, 1, padding='same')(x)
    
    # Reshape output for further processing
    # Actual shape processing would depend on specific YOLO implementation
    return x
