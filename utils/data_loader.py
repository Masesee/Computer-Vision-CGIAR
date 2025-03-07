import tensorflow as tf
import os
from sklearn.model_selection import train_test_split

# Set dataset path
DATASET_PATH = '/kaggle/input/cgiar-root-volume-estimation-challenge/data'

def load_data(validation_split = 0.2):
    train_dir = os.path.join(DATASET_PATH, "train") # The images are contained in subfolders within this main folder

    # Get all images and their corresponding labels
    all_image_paths = []
    all_image_labels = []
    class_names = sorted([d for d in os.path.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(train_dir, class_name)
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_file)
                all_image_paths.append(img_path)
                all_image_labels.append(class_idx)

    # Split the data into train and validation data
    train_paths, val_paths, train_labels, val_labels = train_test_split(all_image_paths, all_image_labels, test_size=validation_split, 
                                                                        stratify=all_image_labels, random_state=42)

    # Function to load and preprocess the images
    def preprocess_image(image_path, label):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [224,224])
        img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0,1]
        # Conver label to one-hot
        label = tf.one_hot(label, depth=len(class_names))
        return img, label
    
    # Create tensorflow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_paths,train_labels))
    train_dataset = train_dataset.map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_paths,val_labels))
    val_dataset = val_dataset.map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)

    # Add class_names attribute to the datasets
    train_dataset.class_names = class_names
    val_dataset.class_names = class_names
    
    return train_dataset, val_dataset
