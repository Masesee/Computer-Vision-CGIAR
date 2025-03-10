from transformers import TFViTModel
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Create custom layers instead of Lambda functions
class PreprocessingLayer(Layer):
    def __init__(self, **kwargs):
        super(PreprocessingLayer, self).__init__(**kwargs)
    
    def call(self, images):
        # Resize if needed
        images = tf.image.resize(images, (224, 224))
        # Normalize pixel values
        images = tf.cast(images, tf.float32) / 255.0
        # Convert from [batch, height, width, channels] to [batch, channels, height, width]
        images = tf.transpose(images, [0, 3, 1, 2])
        return images
    
    def get_config(self):
        config = super(PreprocessingLayer, self).get_config()
        return config

class ViTEncoderLayer(Layer):
    def __init__(self, **kwargs):
        super(ViTEncoderLayer, self).__init__(**kwargs)
        # Load pre-trained ViT model in the initialization
        self.vit_model = TFViTModel.from_pretrained("google/vit-base-patch16-224")
        # Make sure the vit_model is not trainable (optional)
        self.vit_model.trainable = False
    
    def call(self, x):
        outputs = self.vit_model(x)
        # Get the [CLS] token representation
        return outputs.last_hidden_state[:, 0, :]
    
    def get_config(self):
        config = super(ViTEncoderLayer, self).get_config()
        return config

def build_vit(input_shape, num_classes=1, regression=True):
    """
    Build a Vision Transformer (ViT) model for regression using TensorFlow/Keras.
    Uses custom layers instead of Lambda functions for better serialization.
    
    Args:
        input_shape (tuple): Shape of input images (e.g., (224, 224, 3)).
        num_classes (int): Output units (should be 1 for regression).
        regression (bool): Whether to use regression (default: True).
    Returns:
        model: A TensorFlow/Keras ViT model.
    """
    # Define input layer
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_layer")
    
    # Apply preprocessing using custom layer
    preprocessed = PreprocessingLayer(name="preprocessing_layer")(inputs)
    
    # Apply ViT encoder using custom layer
    features = ViTEncoderLayer(name="vit_encoder_layer")(preprocessed)
    
    # Regression head
    output = tf.keras.layers.Dense(num_classes, activation="linear", name="output_layer")(features)
    
    # Build final model
    model = tf.keras.Model(inputs=inputs, outputs=output)
    
    return model

# Function to save a model with custom_objects
def save_vit_model(model, save_path):
    """
    Save the ViT model with custom objects properly registered.
    """
    # Define custom_objects dictionary
    custom_objects = {
        'PreprocessingLayer': PreprocessingLayer,
        'ViTEncoderLayer': ViTEncoderLayer
    }
    
    # Save the model
    model.save(save_path, save_format='h5')
    print(f"Model saved to {save_path}")
    
    return save_path

# Function to load the model with custom objects
def load_vit_model(model_path):
    """
    Load the ViT model with custom objects properly registered.
    """
    # Define custom_objects dictionary
    custom_objects = {
        'PreprocessingLayer': PreprocessingLayer,
        'ViTEncoderLayer': ViTEncoderLayer
    }
    
    # Load the model
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"Model loaded from {model_path}")
    
    return model
