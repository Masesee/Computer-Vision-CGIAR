import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from utils.data_loader import load_data
from tensorflow.keras.models import load_model
from models.resnet import build_resnet
from models.efficientnet import build_efficientnet
from models.mobilenet import build_mobilenet
from models.vit import build_vit

def load_training_history(model_name):
    history_path = f"/kaggle/working/Computer-Vision-CGIAR/{model_name}_history.json"
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            return json.load(f)
    else:
        print(f"No history file found for {model_name}")
        return None

def load_trained_model(model_name, input_shape, num_classes):
    model_path = f"/kaggle/working/Computer-Vision-CGIAR/{model_name}_model.keras"
    
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        print(f"No saved model found for {model_name}")
        return None

def evaluate_models():
    """
    Evaluates all trained models and selects the best one based on validation accuracy.
    Saves the best model's details in a JSON file.
    """
    model_names = ['resnet', 'efficientnet', 'mobilenet', 'vit']
    model_performances = {}

    # Load validation data
    _, val_data = load_data(validation_split=0.2)
    input_shape = (224, 224, 3)
    num_classes = 1

    histories = {}
    
    for model_name in model_names:
        print(f"\nEvaluating {model_name}...")
        history = load_training_history(model_name)
        if history:
            histories[model_name] = history
            
        model = load_trained_model(model_name, input_shape, num_classes)
        if model:
            # Evaluate on validation set
            eval_result = model.evaluate(val_data, verbose=1)
            val_accuracy = eval_result[1]  # Assuming second metric is accuracy
            model_performances[model_name] = {
                'path': f"/kaggle/working/Computer-Vision-CGIAR/{model_name}_model.keras",
                'accuracy': val_accuracy
            }
            print(f"{model_name} Validation Accuracy: {val_accuracy:.4f}")
            
            # Plot regression predictions
            plot_regression_results(model, val_data, model_name)

    # Select the best model (lowest MAE)
    if model_performances:
        best_model_name = min(model_performances, key=lambda k: model_performances[k]['mae'])
        best_model_path = model_performances[best_model_name]['path']
        best_mae = model_performances[best_model_name]['mae']

        print(f"\nBest model: {best_model_name} with Validation MAE: {best_mae:.4f}")

        # Save best model info
        best_model_info = {
            'name': best_model_name,
            'path': best_model_path,
            'mae': float(best_mae)
        }

        json_path = "/kaggle/working/Computer-Vision-CGIAR/best_model_info.json"
        with open(json_path, 'w') as f:
            json.dump(best_model_info, f)

        print(f"Best model info saved to {json_path}")

    if histories:
        plot_mae(histories)
        plot_loss(histories)
    else:
        print("No training histories found. Make sure to save training history during training.")

def plot_mae(histories):
    """Plot Mean Absolute Error (MAE) for multiple models over training epochs."""
    plt.figure(figsize=(12, 6))
    for model_name, history in histories.items():
        if history and 'val_mae' in history: 
            plt.plot(history['val_mae'], label=f'{model_name} Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Model Performance Comparison - Validation MAE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('/kaggle/working/Computer-Vision-CGIAR/validation_mae.png')
    plt.show()

def plot_loss(histories):
    """Plot validation loss for multiple models."""
    plt.figure(figsize=(12, 6))
    for model_name, history in histories.items():
        if history and 'val_loss' in history:  # ✅ Ensure 'val_loss' exists
            plt.plot(history['val_loss'], label=f"{model_name} Validation Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Model Performance Comparison - Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('/kaggle/working/Computer-Vision-CGIAR/validation_loss.png')
    plt.show()

def plot_regression_results(model, dataset, model_name):
    """Plot predicted vs actual root volume for regression."""
    y_true = []
    y_pred = []

    for images, labels in dataset:
        predictions = model.predict(images)
        y_true.extend(labels.numpy().flatten())  # Convert tensor to list
        y_pred.extend(predictions.flatten())  # Flatten predictions

    # Scatter plot: Actual vs. Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, label="Predicted vs. Actual")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color="red", linestyle="--", label="Ideal Prediction")
    plt.xlabel("Actual Root Volume")
    plt.ylabel("Predicted Root Volume")
    plt.title(f"Regression Predictions for {model_name}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(f"/kaggle/working/Computer-Vision-CGIAR/regression_plot_{model_name}.png")
    plt.show()

if __name__ == "__main__":
    evaluate_models()
