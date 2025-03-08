from transformers import TFAutoModel
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
    vit_model = TFAutoModel.from_pretrained("google/vit-base-patch16-224")

    # Define input layer (Standard 224x224 RGB image)
    inputs = tf.keras.layers.Input(shape=input_shape)

    #  Convert to the format required by ViT: (batch_size, 3, 224, 224)
    pixel_values = tf.keras.layers.Permute((3, 1, 2))(inputs)  # Convert (H, W, C) → (C, H, W)

    #  Use Lambda() to cast into tf.float32 (avoids KerasTensor issue)
    pixel_values = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(pixel_values)

    #  Pass processed inputs to ViT model
    vit_outputs = vit_model(pixel_values).last_hidden_state[:, 0, :]

    # Regression head
    output = tf.keras.layers.Dense(1, activation="linear")(vit_outputs)  #  Single neuron for regression

    # Build final model
    model = tf.keras.Model(inputs=inputs, outputs=output)

    return model
