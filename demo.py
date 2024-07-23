from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import os, random, sys
from create_datagenerators import create_data_generators
import json
from sklearn.metrics import confusion_matrix
from data_preprocessing import resize_white, resize_black
from tensorflow.keras.utils import plot_model
from collections import Counter

SAVE_RESULRS_DIR = 'saved_models/'
RESULTS_FOLDER = SAVE_RESULRS_DIR + '/2020'
TRAIN_DIR = 'DATASETS/train'
TEST_DIR ='DATASETS/test'
TRAIN_DIR_TST = TRAIN_DIR
TEST_DIR_TST = TEST_DIR


def load_image(img_path, input_shape, resize=False):
    if resize:
        img, _ = resize_black(input_shape[1], img_path, print_oldsize=False)
    else:
        img = image.load_img(img_path, target_size=input_shape)
    
    img_tensor = image.img_to_array(img)                
    img_tensor = np.expand_dims(img_tensor, axis=0)     
    img_tensor /= 255.0                                

    return img_tensor

def decode_predictions(preds, class_names, top=5):
    results = []
    for pred in preds:
        top_indices = np.argsort(pred)[-top:][::-1]
        top_predictions = [(class_names[i], pred[i]) for i in top_indices]
        results.append(sorted(top_predictions, key=lambda x: x[1], reverse=True))
    return results

def predict(img_path, model, input_shape, class_names, correct_class):
    img_array = load_image(img_path, input_shape)
    preds = model.predict(img_array)
    predictions = decode_predictions(preds, class_names)
    
    # Get the top prediction
    top1_pred = predictions[0][0]
    
    img_org = image.load_img(img_path)
    
    # Plot the results
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img_org)
    axs[0].set_title(f"Correct Class: {correct_class}")
    axs[0].axis('off')
    
    axs[1].imshow(img_array[0])
    axs[1].set_title(f"Prediction: {top1_pred[0]} ({top1_pred[1]:.2f})")
    axs[1].axis('off')
    
    plt.show()

def load_model(results_folder, show_accuracy=False):
    json_path = os.path.join(results_folder, 'model.json')
    weights_path = os.path.join(results_folder, 'weights.best.hdf5')
    with open(json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights(weights_path)

    input_shape = loaded_model.layers[0].input_shape[1:3]
    
    print("Loaded model from disk")
    return loaded_model, input_shape

def perform_pred(car_class, results_folder=RESULTS_FOLDER, test_dir=TEST_DIR_TST, img_pth=None):
    hyperparams_file = os.path.join(results_folder, 'hyperparams.json')
    class_names_file = os.path.join(results_folder, 'class_names.txt')
    with open(hyperparams_file, 'r') as file:
        hyperparams = json.load(file)['hyperparameters'][0]
    batch_size = hyperparams['BATCHSIZE']

    if img_pth is None:
        test_img_dir = os.path.join(test_dir, car_class)
        test_img = os.path.join(test_img_dir, random.choice(os.listdir(test_img_dir)))
    else:
        test_img = img_pth

    loaded_model, input_shape = load_model(results_folder)
    
    if os.path.exists(class_names_file):
        with open(class_names_file, 'r') as file:
            class_names = [line.strip() for line in file]
    else:
        generator_train, _ = create_data_generators(input_shape, batch_size, TRAIN_DIR, TEST_DIR)
        class_names = list(generator_train.class_indices.keys())
        with open(class_names_file, 'w') as file:
            file.write('\n'.join(class_names))
    predict(test_img, loaded_model, input_shape, class_names, car_class)

if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) == 1:
        print('Too few arguments.')
    elif len(sys.argv) == 2:
        perform_pred(sys.argv[1])
    elif len(sys.argv) == 3:
        if str(sys.argv[2]).endswith('.jpg') or str(sys.argv[2]).endswith('.png'):
            perform_pred(sys.argv[1], img_pth=sys.argv[2])
        else:
            perform_pred(sys.argv[1], results_folder=sys.argv[2])
    elif len(sys.argv) == 4:
        if str(sys.argv[3]).endswith('.jpg') or str(sys.argv[3]).endswith('.png'):
            perform_pred(sys.argv[1], results_folder=sys.argv[2], 
                            img_pth=sys.argv[3])
        else:
            perform_pred(sys.argv[1], results_folder=sys.argv[2], 
                                    test_dir=sys.argv[3])
    else:
        print('Too much arguments.')
