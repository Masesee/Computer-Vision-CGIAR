import argparse
from tensorflow.keras.preprocessing import image
import numpy as np
from models.resnet import build_resnet

def load_model():
    model = build_resnet((224, 224, 3), 10)
    model.load_weights('results/model_checkpoints/best_model.h5')
    return model

def predict(image_path):
    model = load_model()
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    return predictions.argmax()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make a prediction')
    parser.add_argument('image_path', type=str, help='Path to image')
    args = parser.parse_args()
    prediction = predict(args.image_path)
    print(f'Predicted Class: {prediction}')
