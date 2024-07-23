# imports
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import numpy as np
import os
import json
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.optimizers import Adam

def set_trainable_layers(model, num_layers_to_freeze):
    """Freeze the first 'num_layers_to_freeze' layers of the model."""
    for layer in model.layers[:num_layers_to_freeze]:
        layer.trainable = False

def finetune_inceptionv3(base_model, transfer_layer, x_trainable, dropout, fc_layers, num_classes, new_weights=""):
    """Fine-tunes the InceptionV3 model."""
    total_layers = len(base_model.layers)
    
    if x_trainable == "all":
        freeze_count = 0
    elif isinstance(x_trainable, int):
        freeze_count = total_layers - x_trainable
    else:
        freeze_count = total_layers
    
    set_trainable_layers(base_model, freeze_count)
    
    # Print layer information
    print(f"Number of all layers in the feature extractor part of model: {total_layers}.")
    print(f"Number of frozen (untrainable) layers in the feature extractor part of model: {freeze_count}.")
    
    # Build the classification part of the model
    x = transfer_layer.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(dropout)(x)
    x = Dense(fc_layers[0], activation='relu')(x)
    x = Dropout(dropout)(x)
    
    for fc_units in fc_layers[1:]:
        x = Dense(fc_units, activation='relu')(x)
        x = Dropout(dropout)(x)
    
    predictions = Dense(num_classes, activation='softmax')(x)
    
    finetune_model = Model(inputs=base_model.input, outputs=predictions)
    
    # Load weights if provided
    if new_weights:
        finetune_model.load_weights(new_weights)
    
    return finetune_model

if __name__ == "__main__":
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
    transfer_layer = base_model.get_layer(index=-1)
    
    print(transfer_layer)
    
    new_model = finetune_inceptionv3(
        base_model=base_model,
        transfer_layer=transfer_layer,
        x_trainable=0,  # Adjust as needed
        dropout=0.5,
        fc_layers=[1024, 1024],
        num_classes=196
    )
    
    optimizer = Adam(learning_rate=0.000001)
    new_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(new_model.summary())
    for layer in new_model.layers:
        print(layer, layer.trainable)
