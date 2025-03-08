import argparse
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Model path to match evaluate_models.py output
BEST_MODEL_PATH = "/kaggle/working/Computer-Vision-CGIAR/best_model_info.json"

def load_best_model():
    """Loads the best trained model based on saved evaluation results."""
    import json
    import os

    # Load best model details from JSON
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"Best model info file not found: {BEST_MODEL_PATH}")

    with open(BEST_MODEL_PATH, 'r') as f:
        best_model_info = json.load(f)

    best_model_path = best_model_info.get("path")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model file not found: {best_model_path}")

    print(f"Loading best model: {best_model_path}")
    return load_model(best_model_path)

def predict(image_path):
    """Preprocess image and predict root volume."""
    model = load_best_model()
    
    # Ensure model expects a single output for regression
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    
    return float(prediction[0][0])  # Ensure output is a single float value

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make a prediction')
    parser.add_argument('image_path', type=str, help='Path to image')
    args = parser.parse_args()

    prediction = predict(args.image_path)
    print(f'Predicted Root Volume: {prediction:.4f}')
