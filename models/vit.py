from transformers import TFViTModel
import tensorflow as tf

def build_vit(input_shape, num_classes=1, regression=True):
    """
    Build a Vision Transformer (ViT) model for regression using TensorFlow/Keras.
    Args:
        input_shape (tuple): Shape of input images (e.g., (224, 224, 3)).
        num_classes (int): Output units (should be 1 for regression).
        regression (bool): Whether to use regression (default: True).
    Returns:
        model: A TensorFlow/Keras ViT model.
    """
    # Define input layer (Standard 224x224 RGB image)
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Create a custom preprocessing layer to handle the input transformation
    def preprocess_images(images):
        # Resize if needed
        images = tf.image.resize(images, (224, 224))
        # Normalize pixel values
        images = tf.cast(images, tf.float32) / 255.0
        # Convert from [batch, height, width, channels] to [batch, channels, height, width]
        images = tf.transpose(images, [0, 3, 1, 2])
        return images
    
    # Apply preprocessing
    preprocessed = tf.keras.layers.Lambda(preprocess_images)(inputs)
    
    # Load pre-trained ViT model
    vit_model = TFViTModel.from_pretrained("google/vit-base-patch16-224")
    
    # Use the functional API to create a wrapper around the Hugging Face model
    def vit_encoder(x):
        outputs = vit_model(x)
        # Get the [CLS] token representation
        return outputs.last_hidden_state[:, 0, :]
    
    # Apply the ViT model through a Lambda layer
    features = tf.keras.layers.Lambda(vit_encoder)(preprocessed)
    
    # Regression head
    output = tf.keras.layers.Dense(num_classes, activation="linear")(features)
    
    # Build final model
    model = tf.keras.Model(inputs=inputs, outputs=output)
    
    return model
