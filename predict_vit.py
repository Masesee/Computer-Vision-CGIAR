import argparse
import numpy as np
import os
import pandas as pd
import re
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing import image
from transformers import TFViTModel
from tensorflow.keras.layers import Layer, Lambda
from tensorflow import keras

# Paths
MAIN_PATH = Path('/kaggle/input/cgiar-root-volume-estimation-challenge')
DATASET_PATH = MAIN_PATH / "data"

# Define the preprocessing function explicitly
def preprocess_images(images):
    # Resize if needed
    images = tf.image.resize(images, (224, 224))
    # Normalize pixel values
    images = tf.cast(images, tf.float32) / 255.0
    # Convert from [batch, height, width, channels] to [batch, channels, height, width]
    images = tf.transpose(images, [0, 3, 1, 2])
    return images

# Define the custom layers needed for the ViT model
class PreprocessingLayer(Layer):
    def __init__(self, **kwargs):
        super(PreprocessingLayer, self).__init__(**kwargs)
    
    def call(self, images):
        return preprocess_images(images)
    
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

# Updated for .keras format compatibility
def load_vit_model(model_path):
    """
    Load a ViT model with custom layers properly registered.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        model: The loaded TensorFlow model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Define custom_objects dictionary with the preprocessing function
    custom_objects = {
        'PreprocessingLayer': PreprocessingLayer,
        'ViTEncoderLayer': ViTEncoderLayer,
        'preprocess_images': preprocess_images,  # Added the missing function
        'Lambda': Lambda  # Ensure Lambda layer is recognized
    }
    
    try:
        # First try loading with custom objects
        print(f"Loading ViT model from {model_path} with custom layers...")
        
        # Handle different file extensions
        if model_path.endswith('.keras'):
            # Load using Keras 3.x API
            model = keras.models.load_model(model_path, custom_objects=custom_objects)
        else:
            # For older h5 format
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            
        print("ViT model loaded successfully.")
    except Exception as e:
        print(f"Custom loader failed: {str(e)}")
        print("Attempting to load with default loader...")
        
        # Reconstructing the model architecture due to load failure
        print("Attempting to reconstruct the model architecture...")
        try:
            # This is a simplified architecture - you may need to adjust based on your actual model
            input_layer = tf.keras.layers.Input(shape=(224, 224, 3))
            
            # Use PreprocessingLayer instead of Lambda
            x = PreprocessingLayer()(input_layer)
            
            # ViT encoder
            x = ViTEncoderLayer()(x)
            
            # Add final prediction layers
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            output = tf.keras.layers.Dense(1)(x)
            
            model = tf.keras.Model(inputs=input_layer, outputs=output)
            
            # Load weights only (not full architecture)
            try:
                model.load_weights(model_path)
                print("Successfully reconstructed model and loaded weights.")
            except:
                print("Warning: Could not load weights into reconstructed model.")
                
        except Exception as reconstruct_error:
            print(f"Model reconstruction failed: {str(reconstruct_error)}")
            # Final fallback
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully with default loader.")
    
    return model

def predict_single_image(image_path, model):
    """
    Preprocess a single image and predict root volume.
    
    Args:
        image_path (str): Path to the image file
        model: The loaded model to use for prediction
        
    Returns:
        float: Predicted root volume
    """
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array, verbose=0)
        
        return float(prediction[0][0])  # Ensure output is a single float value
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def get_images_within_range(base_path, folder, side, start, end):
    """
    Get images from a folder that match the specified side (L/R) and layer range.
    
    Args:
        base_path (Path): Base path to search for images
        folder (str): Folder name
        side (str): Side identifier ('L' or 'R')
        start (int): Start layer number
        end (int): End layer number
        
    Returns:
        list: List of image paths that match the criteria
    """
    folder_path = Path(base_path) / folder
    
    # Check if folder exists
    if not folder_path.exists():
        return []
    
    # Regex pattern to match filenames (e.g., A6dzrkjqvl_L_033.png)
    pattern = re.compile(r'_([LR])_(\d{3})\.png$')
    
    # Select images within range
    selected_images = []
    for img_name in os.listdir(folder_path):
        match = pattern.search(img_name)
        if match:
            img_side = match.group(1)
            layer = int(match.group(2))
            if img_side == side and start <= layer <= end:
                selected_images.append(str(folder_path / img_name))
    
    return selected_images

def generate_predictions(model_path, output_path):
    """
    Generate predictions using the loaded ViT model.
    
    Args:
        model_path (str): Path to the model file
        output_path (str): Path to save the predictions CSV
    """
    # Load the model
    model = load_vit_model(model_path)
    
    # Load test metadata
    test_csv_path = MAIN_PATH / "Test.csv"
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test CSV file not found: {test_csv_path}")
    
    test_df = pd.read_csv(test_csv_path)
    print(f"Loaded {len(test_df)} test samples from {test_csv_path}")
    
    # Prepare results dataframe
    results = []
    
    # Process each test sample
    for index, row in test_df.iterrows():
        sample_id = row["ID"]
        folder = row["FolderName"]
        side = row["Side"]
        start_layer = row["Start"]
        end_layer = row["End"]
        
        # Get all relevant images
        image_paths = get_images_within_range(DATASET_PATH / "test", folder, side, start_layer, end_layer)
        
        if not image_paths:
            print(f"Warning: No images found for sample {sample_id}")
            results.append({
                "ID": sample_id,
                "RootVolume": 0  # Default value when no images found
            })
            continue
        
        # Predict root volume for each image
        predictions = []
        for img_path in image_paths:
            prediction = predict_single_image(img_path, model)
            if prediction is not None:
                predictions.append(prediction)
        
        # Calculate average prediction if we have valid predictions
        if predictions:
            avg_prediction = sum(predictions) / len(predictions)
        else:
            avg_prediction = 0  # Default value when predictions fail
        
        # Add to results
        results.append({
            "ID": sample_id,
            "RootVolume": avg_prediction
        })
        
        # Progress indicator
        if (index + 1) % 10 == 0 or (index + 1) == len(test_df):
            print(f"Processed {index + 1}/{len(test_df)} samples")
    
    # Create and save final predictions dataframe
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Saved predictions for {len(results_df)} samples to {output_path}")
    
    return results_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate predictions using a ViT model for CGIAR root volume challenge')
    parser.add_argument('--model', type=str, required=True, help='Path to the ViT model to use for prediction')
    parser.add_argument('--output', type=str, default="/kaggle/working/vit_submission.csv", 
                        help='Path to save predictions CSV')
    
    args = parser.parse_args()
    generate_predictions(args.model, args.output)
