import tensorflow as tf
import pandas as pd
import os
import re
from pathlib import Path
from sklearn.model_selection import train_test_split

# Set dataset path
DATASET_PATH = Path('/kaggle/input/cgiar-root-volume-estimation-challenge/data')
MAIN_PATH = Path('/kaggle/input/cgiar-root-volume-estimation-challenge')

def get_images_within_range(base_path: Path, folder: str, side: str, start: int, end: int):
    """
    Get images from a folder that match the specified side (L/R) and layer range.

    Args:
        base_path (Path): Root directory containing all folders.
        folder (str): Name of the target folder.
        side (str): Scan side to filter ('L' or 'R').
        start (int): Starting layer (inclusive).
        end (int): Ending layer (inclusive).

    Returns:
        list[Path]: List of matching image file paths.
    """
    folder_path = base_path / folder

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
                selected_images.append(folder_path / img_name)

    return selected_images

def preprocess_image(image_path):
    """Loads and preprocesses an image for the model."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)  # PNG format
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0  # Normalize
    return img

def load_data(validation_split=0.2):
    """
    Load images and corresponding root volume labels.

    Args:
        validation_split (float): Percentage of training data to use for validation.

    Returns:
        train_dataset, val_dataset: TensorFlow datasets for training and validation.
    """
    # Load CSV with metadata
    csv_path = MAIN_PATH / "Train.csv"
    df = pd.read_csv(csv_path)

    image_paths = []
    labels = []

    # Iterate through each row to retrieve corresponding images
    for _, row in df.iterrows():
        folder = row["FolderName"]
        side = row["Side"]
        start_layer = row["Start"]
        end_layer = row["End"]
        root_volume = row["RootVolume"]

        # Retrieve relevant image paths
        images = get_images_within_range(DATASET_PATH / "train", folder, side, start_layer, end_layer)

        if images:
            image_paths.extend(images)
            labels.extend([root_volume] * len(images))  # Same label for all images

    # Split into training & validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=validation_split, random_state=42
    )

    def preprocess(image_path, label):
        img = preprocess_image(image_path)
        return img, tf.expand_dims(label, axis=-1)  # Ensure shape (batch, 1)

    # Convert to TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_dataset = train_dataset.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_dataset = val_dataset.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset

# def load_data(validation_split = 0.2, task="regression"):
#     csv_path = os.path.join(MAIN_PATH, "Train.csv")
#     df = pd.read_csv(csv_path)

#     # Extract image file paths and labels
#     image_paths = 

    
#     train_dir = os.path.join(DATASET_PATH, "train") # The images are contained in subfolders within this main folder

#     # Get all images and their corresponding labels
#     all_image_paths = []
#     all_image_labels = []
#     class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

#     for class_idx, class_name in enumerate(class_names):
#         class_dir = os.path.join(train_dir, class_name)
#         for img_file in os.listdir(class_dir):
#             if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 img_path = os.path.join(class_dir, img_file)
#                 all_image_paths.append(img_path)
#                 all_image_labels.append(class_idx)

#     # Split the data into train and validation data
#     train_paths, val_paths, train_labels, val_labels = train_test_split(all_image_paths, all_image_labels, test_size=validation_split, 
#                                                                         stratify=all_image_labels, random_state=42)

#     # Function to load and preprocess the images
#     def preprocess_image(image_path, label):
#         img = tf.io.read_file(image_path)
#         img = tf.image.decode_image(img, channels=3, expand_animations=False)
#         img = tf.image.resize(img, [224,224])
#         img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0,1]
#         # Conver label to one-hot
#         label = tf.one_hot(label, depth=len(class_names))
#         return img, label
    
#     # Create tensorflow datasets
#     train_dataset = tf.data.Dataset.from_tensor_slices((train_paths,train_labels))
#     train_dataset = train_dataset.map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)

#     val_dataset = tf.data.Dataset.from_tensor_slices((val_paths,val_labels))
#     val_dataset = val_dataset.map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)

#     # Add class_names attribute to the datasets
#     train_dataset.class_names = class_names
#     val_dataset.class_names = class_names
    
#     return train_dataset, val_dataset
