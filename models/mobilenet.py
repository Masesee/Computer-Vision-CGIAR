from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

def build_mobilenet(input_shape, num_classes=1, regression=True):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    output = Dense(num_classes, activation='linear')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model
