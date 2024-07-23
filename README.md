Car Classification Using Deep Learning

The project aims to develop an entrance gate system capable of fully recognizin an approaching car. The car recognition process involves three key tasks:

1. Identifying the make and model of the car based on its shape.
2. Recognizing the company logo.
3. Reading and interpreting the car's number plate.

The classification algorithm utilizes transfer learning and fine-tuning of the Inception-v3 network [1][2] with the Cars Dataset from Stanford [3]. The Keras implementation of Inception-v3 was employed [4]. The final model can recognize 195different classes of cars with an overall accuracy of 81%. Each class name includes the company name, model, type, and year of production.

### Training Your Own Model

#### 1. Prepare a Dataset

**a) Using the Stanford Dataset:**
- The main folder should have two subfolders: `train` and `test`.
- Each subfolder should contain more subdirectories, each named after a class (label) (one subdirectory per class).
- You can prepare the data yourself using the original Cars Stanford Dataset or download the pre-sorted dataset from [here](#).

- To modify images (resize, expand background, transform to grayscale), refer to the functions inside `data_preprocessing.py`.

**b) Using a New Dataset:**
- Prepare the dataset by saving images inside the appropriate subdirectories, each named after the class (label).

#### 2. Modify `hyperparams.json`
Adjust the hyperparameters and other training settings in the `hyperparams.json` file:

- **WEIGHTS**: Initial weights
- **EPOCHS**: Number of epochs to train
- **BATCHSIZE**: Number of training batches
- **LEARN_RATE**: Learning rate
- **DROPOUT**: Dropout rate
- **TRAIN_LAYERS**: Number of trainable layers. Set to 0 to train only the classifier; set to “all” to train all layers. To set the last x layers as trainable, set this parameter to “x”.
- **FC_LAYERS**: Shape of fully connected layers in the classifier

#### 3. Check Paths
Verify the hardcoded paths in the script, which lead to:

- Datasets: `TRAIN_DIR`, `TEST_DIR`
- Settings file: `hyperparams.json`
- Results directory: `RESULTS_FOLDER` (default: `saved_models/`)

#### 4. Run the Training
Execute the training script

### Results Folder
The results folder (containing weights, model structure, accuracy plots, etc.) will be saved automatically inside `saved_models/xxxxxxxx_xxxx` (where `xxxxxxxx_xxxx` is the date and time the training started). Depending on the dataset size and your machine's computational power, the training process may take from a few hours to a few days (Stanford Dataset on a GPU: ~24 hours).

### Analyze Results
To analyze the results (confusion matrix, accuracy for each class, and more statistics on model performance), use `analyse_results.ipynb`. Specify the paths for:

- Datasets: `TRAIN_DIR`, `TEST_DIR`
- Settings file: `hyperparams.json`
- Results directory: `RESULTS_FOLDER` (default: `saved_models/2020`)
