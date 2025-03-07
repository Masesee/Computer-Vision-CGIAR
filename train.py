import argparse
from models.resnet import build_resnet
from models.efficientnet import build_efficientnet
from models.mobilenet import build_mobilenet
from models.vit import build_vit
from utils.data_loader import load_data

def main():
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument('--model', type=str, required=True, choices=['resnet', 'efficientnet', 'mobilenet', 'vit'], help='Choose model')
    args = parser.parse_args()

    input_shape = (224, 224, 3)
    train_data, val_data = load_data()
    num_classes = len(train_data.class_names) # Get the correct number of classes
    
    models = {
        "resnet": build_resnet,
        "efficientnet": build_efficientnet,
        "mobilenet": build_mobilenet,
        "vit": build_vit
    }

    model = models[args.model](input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, validation_data=val_data, epochs=10)

if __name__ == '__main__':
    main()
