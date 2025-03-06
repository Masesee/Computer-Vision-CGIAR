import matplotlib.pyplot as plt

def plot_accuracy(histories):
    plt.figure(figsize=(10, 5))
    for model_name, history in histories.items():
        plt.plot(history.history['val_accuracy'], label=f'{model_name} Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title('Model Performance Comparison')
    plt.legend()
    plt.show()
