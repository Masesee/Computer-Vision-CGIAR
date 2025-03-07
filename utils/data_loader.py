import tensorflow as tf
import os

# Set dataset path
DATASET_PATH = '/kaggle/input/cgiar-root-volume-estimation-challenge'

def load_data():
    train_dir = os.path.join(DATASET_PATH, "train") # The images are contained in subfolders within this main folder
    val_dir = os.path.join(DATASET_PATH, "test") # The images are contained in subfolders within this main folder
    
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir, image_size=(224, 224), batch_size=32)
    val_data = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir, image_size=(224, 224), batch_size=32)
    return train_data, val_data
