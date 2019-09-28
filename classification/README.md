# Medical Image Classification

This classification codebase was designed to provide a convenient pipeline to take a set of medical images and train a variety of convolutional neural networks (CNN) to classify those images in Pytorch. Features include:

- Automated data loading, training, and model evaluation
- Implementation of many common CNNs
- Support for 2d, 2.5d, and 3d medical image classification
- Support for weak supervision using text reports
- Training monitoring via tensorboardX

## Installation 

*Note: Codebase only supports Python 3.*

1. Create a virtual environment, within which we'll install the codebase and its dependencies:
```
pip install virtualenv
virtualenv -p python3.6 medImgEnv
```

2. Activate the virtual environment we just created:
```
source medImgEnv/bin/activate
```

3. Clone the repository:
```
git clone 
```

4. Install the required Python dependencies: 
```
pip install -r requirements.txt
```

## Getting started
Required files, details on the codebase organization, and examples of how to use this codebase to classify medical images are provided in the Jupyter notebookes __tutorials/headCTclassifier.ipynb__. To enable function calls from the command line, __run_DICOM_crawler.py__ is also provided, which perform the same operations as their respective Jupyter notebooks listed above.

## Example usage
To train a classifier over a set of images stored in pixel_data_example.h5 with labels stored in train_example.csv, validation_example.csv, and test_example.csv, run:
```
python run_classifier.py ....
```
This command will train a 2d resnet18 to classify the images stored in pixel_data_example.h5 according to the labels in train.csv and valid.csv. It will then evaluate the model according to the labels in test.csv. Files generated from this command include: train_log.txt, which stores all outputs from the training process; predicted_labels.csv, which are the best trained model's predictions on the test set (if provided, otherwise the validation set); learning_curve.png, a simple plot of the training and validation loss curves throughout training; a tensorboard events file, which you can use to view all performance metrics and losses as a function of training the train and validation sets; model_best_train_loss.pth and model_best_val_loss.pth, which contain the stored models that acheived the best loss on the trian and validation set, respectively; and finally all_params.txt, which contains the configuration used this run.

To view all parameter options for training, run:

```
python run_classifier.py -h
```
