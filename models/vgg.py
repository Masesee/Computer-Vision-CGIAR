from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

def build_vgg(input_shape, num_classes=1, regression=False):
    # Load pre-trained VGG16 model without top layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Add custom top layers
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    
    # Output layer - regression or classification
    activation = "linear" if regression else "softmax"
    output = Dense(num_classes, activation=activation)(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=output)
    
    return model
