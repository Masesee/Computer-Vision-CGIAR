import argparse
import numpy as np
import os
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import glob

# Model path to match evaluate_models.py output
BEST_MODEL_PATH = "/kaggle/working/Computer-Vision-CGIAR/best_model_info.json"

def load_best_model():
    """Loads the best trained model based on saved evaluation results."""
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

def find_all_images(directory_path):
    """Recursively find all image files in directory and subdirectories."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_files = []
    
    # Walk through all directories recursively
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # Check if the file has an image extension
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    return image_files

def predict_directory(directory_path, model):
    """Process all images in a directory and its subdirectories and return predictions."""
    # Find all image files recursively
    image_files = find_all_images(directory_path)
    
    if not image_files:
        print(f"No image files found in {directory_path} or its subdirectories")
        return None
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    results = []
    for img_path in sorted(image_files):
        # Get relative path from the base directory for cleaner output
        rel_path = os.path.relpath(img_path, directory_path)
        prediction = predict_single_image(img_path, model)
        if prediction is not None:
            results.append({
                'image_path': rel_path,
                'predicted_root_volume': prediction
            })
            print(f"Predicted Root Volume for {rel_path}: {prediction:.4f}")
    
    # Create a DataFrame with results
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    output_path = os.path.join(os.path.dirname(directory_path), 'predictions.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Saved {len(results)} predictions to {output_path}")
    
    return results_df

def predict(path):
    """Preprocess image or directory of images and predict root volume."""
    model = load_best_model()
    
    if os.path.isdir(path):
        print(f"Processing directory: {path}")
        return predict_directory(path, model)
    else:
        print(f"Processing single image: {path}")
        prediction = predict_single_image(path, model)
        print(f'Predicted Root Volume: {prediction:.4f}')
        return prediction

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make predictions on images')
    parser.add_argument('image_path', type=str, help='Path to image or directory of images')
    parser.add_argument('--output', type=str, help='Path to save predictions (CSV for directories)', default=None)
    
    args = parser.parse_args()
    result = predict(args.image_path)
    
    # If output path is provided and result is a DataFrame, save to the specified location
    if args.output and isinstance(result, pd.DataFrame):
        result.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")
