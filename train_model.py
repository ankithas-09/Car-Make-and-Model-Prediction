from create_model import finetune_inceptionv3
from create_datagenerators import create_data_generators
import json
import codecs
import numpy as np
import datetime
import shutil
import os
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
HYPERPARAMS_FILE = 'hyperparams.json'
TRAIN_DIR = 'DATASETS/train'
TEST_DIR = 'DATASETS/test'
SAVE_RESULTS_DIR = 'saved_models/'

def load_hyperparameters(file_path):
    """Load hyperparameters from a JSON file."""
    with open(file_path, "r") as file:
        data = json.load(file)
    return data['hyperparameters'][0]

def create_folder_with_results(base_path=SAVE_RESULTS_DIR, access_rights=0o755):
    """Create a timestamped directory for saving models."""
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M")
    directory_path = os.path.join(base_path, timestamp)
    
    try:
        os.makedirs(directory_path, mode=access_rights, exist_ok=True)
        print(f"Directory created: {directory_path}")
        return directory_path
    except OSError as e:
        print(f"Failed to create directory {directory_path}: {e}")
        return None

def save_history(path, history):
    """Save training history to a JSON file."""
    history_data = {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in history.history.items()}
    with codecs.open(path, 'w', encoding='utf-8') as file:
        json.dump(history_data, file, separators=(',', ':'), sort_keys=True, indent=4)

def plot_training_history(history, acc_path, loss_path):
    """Plot and save training and validation accuracy/loss."""
    def plot_metrics(epochs, data, labels, title, ylabel, path):
        plt.figure()
        for metric, label in zip(data, labels):
            plt.plot(epochs, metric, label=label)
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(path)
        plt.close()

    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs = range(len(acc))

    plot_metrics(epochs, [acc, val_acc], ['Training Accuracy', 'Validation Accuracy'], 'Training and Validation Accuracy', 'Accuracy', acc_path)
    plot_metrics(epochs, [loss, val_loss], ['Training Loss', 'Validation Loss'], 'Training and Validation Loss', 'Loss', loss_path)

if __name__ == "__main__":
    hyperparams = load_hyperparameters(HYPERPARAMS_FILE)
    TRAINING_TIME_PATH = create_folder_with_results()

    shutil.copy2(HYPERPARAMS_FILE, TRAINING_TIME_PATH)

    base_model = InceptionV3(weights=hyperparams['WEIGHTS'], include_top=False, input_shape=(299,299,3))
    input_shape = base_model.layers[0].output_shape[1:3]
    transfer_layer = base_model.get_layer(index=-1)

    generator_train, generator_test = create_data_generators(
        input_shape=input_shape, 
        batch_size=hyperparams['BATCHSIZE'], 
        train_dir=TRAIN_DIR, 
        test_dir=TEST_DIR, 
        save_augmented=None, 
        plot_imgs=False
    )

    class_names = list(generator_train.class_indices.keys())
    with open(os.path.join(TRAINING_TIME_PATH, 'class_names.txt'), 'w') as file:
        file.writelines(f"{name}\n" for name in class_names)

    finetune_model = finetune_inceptionv3(
        base_model, transfer_layer, hyperparams['TRAIN_LAYERS'], 
        dropout=hyperparams['DROPOUT'], 
        fc_layers=hyperparams['FC_LAYERS'], 
        num_classes=generator_train.num_classes,
        new_weights=hyperparams['NEW_WEIGHTS']
    )

    optimizer = Adam(learning_rate=hyperparams['LEARN_RATE'])
    finetune_model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    class_weight = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(generator_train.classes), 
        y=generator_train.classes
    )

    checkpoint_path = os.path.join(TRAINING_TIME_PATH, 'weights.best.hdf5')
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    history = finetune_model.fit(
        generator_train,
        epochs=hyperparams['EPOCHS'],
        steps_per_epoch=generator_train.n // hyperparams['BATCHSIZE'],
        class_weight=class_weight,
        validation_data=generator_test,
        validation_steps=generator_test.n // hyperparams['BATCHSIZE'],
        callbacks=[checkpoint],
        verbose=0
    )

    save_history(os.path.join(TRAINING_TIME_PATH, 'history.json'), history)
    with open(os.path.join(TRAINING_TIME_PATH, 'model_summary.txt'), 'w') as file:
        finetune_model.summary(print_fn=lambda x: file.write(x + '\n'))

    plot_training_history(
        history,
        os.path.join(TRAINING_TIME_PATH, 'accuracy_vs_epochs.png'),
        os.path.join(TRAINING_TIME_PATH, 'loss_vs_epochs.png')
    )
