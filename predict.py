import argparse
import numpy as np
import os
import pandas as pd
import re
from pathlib import Path
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Paths
BEST_MODEL_PATH = "/kaggle/working/Computer-Vision-CGIAR/best_model_info.json"
MAIN_PATH = Path('/kaggle/input/cgiar-root-volume-estimation-challenge')
DATASET_PATH = MAIN_PATH / "data"

def load_best_model(model_path=None):
    """
    Loads a model for prediction.
    
    Args:
        model_path (str, optional): Path to a specific model to load.
                                   If None, loads the best model from best_model_info.json
    """
    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Specified model file not found: {model_path}")
        print(f"Loading specified model: {model_path}")
        return load_model(model_path)
    
    # Default behavior: load best model from JSON
    import json
    
    # Load best model details from JSON
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"Best model info file not found: {BEST_MODEL_PATH}")
    
    with open(BEST_MODEL_PATH, 'r') as f:
        best_model_info = json.load(f)
    
    best_model_path = best_model_info.get("path")
    if not best_model_path or not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model file not found: {best_model_path}")
    
    print(f"Loading best model: {best_model_path}")
    return load_model(best_model_path)

def predict_single_image(image_path, model):
    """Preprocess a single image and predict root volume."""
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
    Based on the implementation from data_loader.py.
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

def predict_test_data(model_path=None):
    """
    Process test data and produce formatted predictions according to competition requirements.
    
    Args:
        model_path (str, optional): Path to a specific model to use for prediction.
    """
    model = load_best_model(model_path)
    
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
    output_path = "/kaggle/working/submission.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Saved predictions for {len(results_df)} samples to {output_path}")
    
    return results_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate predictions for CGIAR root volume challenge')
    parser.add_argument('--model', type=str, help='Path to a specific model to use for prediction', default=None)
    parser.add_argument('--output', type=str, help='Path to save predictions CSV', default="/kaggle/working/submission.csv")
    
    args = parser.parse_args()
    result_df = predict_test_data(model_path=args.model)
    
    if args.output != "/kaggle/working/submission.csv":
        result_df.to_csv(args.output, index=False)
        print(f"Results also saved to {args.output}")
