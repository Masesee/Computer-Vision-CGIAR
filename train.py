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
    train_data, val_data = load_data(validation_split=0.2)
    num_classes = len(train_data.class_names) # Get the correct number of classes
    
    models = {
        "resnet": build_resnet,
        "efficientnet": build_efficientnet,
        "mobilenet": build_mobilenet,
        "vit": build_vit
    }

    model = models[args.model](input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train with validation set
    history = model.fit(train_data, validation_data=val_data, epochs=10)

    # Save the model
    model_save_path = f"/kaggle/working/Computer-Vision-CGIAR/{args.model}_model"
    model.save(model_save_path)
    print(f' Model saved to {model_save_path}')

    # Save the training history
    import json
    with open(f"/kaggle/working/Computer-Vision-CGIAR/{args.model}_history.json", 'w') as f:
        json.dump(history.history, f)
    print(f"Training history saved for {args.model}")


if __name__ == '__main__':
    main()
