import tensorflow as tf

def load_data():
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        'datasets/train', image_size=(224, 224), batch_size=32)
    val_data = tf.keras.preprocessing.image_dataset_from_directory(
        'datasets/val', image_size=(224, 224), batch_size=32)
    return train_data, val_data
