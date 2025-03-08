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

    # Define inputs
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Convert image input to ViT expected format
    pixel_values = tf.keras.layers.Rescaling(1.0 / 255)(inputs)  # Normalize images
    vit_outputs = vit_model(pixel_values).last_hidden_state[:, 0, :]

    # Regression head
    output = tf.keras.layers.Dense(1, activation="linear")(vit_outputs)  # Single neuron for regression

    # Build final model
    model = tf.keras.Model(inputs=inputs, outputs=output)

    return model
