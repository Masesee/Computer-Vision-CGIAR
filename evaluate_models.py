import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import json
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from models.resnet import build_resnet
from models.efficientnet import build_efficientnet
from models.mobilenet import build_mobilenet
from models.vit import build_vit
from utils.data_loader import load_data

def load_training_history(model_name):
    history_path = f"/kaggle/working/Computer-Vision-CGIAR/{model_name}_history.json"
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            return json.load(f)
    else:
        print(f"No history file found for {model_name}")
        return None

def load_trained_model(model_name, imput_shape, num_classes):
    model_path = f"/kaggle/working/Computer-Vision-CGIAR/{model_name}_model"
    
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        print(f"No saved model found for {model_name}")
        # Create model architecture (but it won't have trained weights)
        models = {
            "resnet": build_resnet,
            "efficientnet": build_efficientnet,
            "mobilenet": build_mobilenet,
            "vit": build_vit
        }
        
        if model_name in models:
            model = models[model_name](input_shape, num_classes)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return model
        else:
            return None

def plot_accuracy(histories):
    plt.figure(figsize=(12, 6))
    for model_name, history in histories.items():
        if history and 'val_accuracy' in history:
            plt.plot(history['val_accuracy'], label=f'{model_name} Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title('Model Performance Comparison - Validation Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('/kaggle/working/Computer-Vision-CGIAR/validation_accuracy.png')
    plt.show()

def plot_loss(histories):
    """Plot validation loss for multiple models"""
    plt.figure(figsize=(12,6))
    for model_name, history in histories.items():
        if history and 'val_loss' in history:
            plt.plot(history['val_loss'], label=f"{model_name} Validation Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Model Performance Comparison - Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha = 0.7)
    plt.savefig('/kaggle/working/Computer-Vision-CGIAR/validation_loss.png')
    plt.show()

def plot_confusion_matrix(model, dataset, class_names, model_name):
    """Plot confusion matrix for the model on the given dataset"""
    # Get predictions
    y_true = []
    y_pred = []
    
    for images, labels in dataset:
        predictions = model.predict(images)
        y_true.extend(np.argmax(labels, axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))
    
    # Create confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues',
                xticklabels=range(len(class_names)),
                yticklabels=range(len(class_names)))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f'/kaggle/working/Computer-Vision-CGIAR/confusion_matrix_{model_name}.png')
    plt.show()
    
    # Print classification report
    print(f"\nClassification Report - {model_name}")
    print(classification_report(y_true, y_pred, target_names=class_names[:10]))  # Print first 10 classes to save space

def main():
    # Set up models to evaluate
    model_names = ['resnet', 'efficientnet', 'mobilenet', 'vit']
    
    # Load validation data
    _, val_data = load_data(validation_split=0.2)
    input_shape = (224, 224, 3)
    num_classes = len(val_data.class_names)
    
    # Load histories and models
    histories = {}
    for model_name in model_names:
        print(f"\nEvaluating {model_name}...")
        history = load_training_history(model_name)
        if history:
            histories[model_name] = history
            
        model = load_trained_model(model_name, input_shape, num_classes)
        if model:
            # Evaluate on validation set
            print(f"Evaluating {model_name} on validation set:")
            eval_result = model.evaluate(val_data, verbose=1)
            print(f"{model_name} Validation Loss: {eval_result[0]:.4f}")
            print(f"{model_name} Validation Accuracy: {eval_result[1]:.4f}")
            
            # Plot confusion matrix for this model
            plot_confusion_matrix(model, val_data, val_data.class_names, model_name)
    
    # Plot comparison charts if we have histories
    if histories:
        plot_accuracy(histories)
        plot_loss(histories)
    else:
        print("No training histories found. Make sure to save training history during training.")

if __name__ == "__main__":
    main()
