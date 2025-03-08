from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

def build_resnet(input_shape, num_classes=1, regression=False):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    # Regression: Output single continuous value instead of classification
    activation = "linear" if regression else "softmax"
    output = Dense(num_classes, activation=activation)(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model
