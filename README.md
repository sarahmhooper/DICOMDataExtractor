# Medical Image Classification and DICOM Information Extraction

We provide a general codebase to perform medical image classification via deep learning using Pytorch. We split the codebase along its two primary functions: __data extraction from DICOM images__ and __classification__. We have separated the two functions because DICOM information extraction is often needed outside of classification tasks, and vice versa. By supplying each function separately, the DICOM extraction code can be used in any DICOM preprocessing pipeline, and the classification code may be used with any set of images.

## Data Extraction from DICOM Files
Medical images are often stored as DICOMs, which can be an inconvenient storage method for machine learning. The metadata stored in each individual DICOM file is inefficient to mine, while grouping all DICOM files from a single scan and stacking the pixel data into a 3d image volume can be a slow process. To address these problems, we provide a DICOM "crawling" script, which provides the following useful functions: 
- Given a set of DICOM files, the DICOM crawling script organizes all metadata from all provided DICOM files into a single CSV, which can be more conviently mined. 
- Given a set of DICOM files, the script stores all pixel data from all provided DICOMs in an h5 file which can be more efficiently accessed and is more familiar to machine learning practitioners. 
- If desired, the script will automatically identify DICOM files that originated from the same scan, order the slices into a 3d image volume, and save the 3d image in an h5 file.

## Medical Image Classification
This classification codebase was designed to provide a convenient pipeline to take a set of medical images and train a variety of convolutional neural networks (CNN) to classify those images in Pytorch. Features include:

- Automated data loading, training, and model evaluation
- Implementation of many common CNNs
- Support for 2d, 2.5d, and 3d medical image classification
- Support for weak supervision using text reports
- Training monitoring via tensorboardX

This project is ongoing, and we'll continue to add to and adjust the codebase over time. Any comments, suggestions, or additions are welcome!

## Installation 

*Note: Codebase only supports Python 3.*

Installation instructions are provided in data_extraction/README.md and classification/README.md.

## Getting started
Required files, details on the codebase organization, and examples of how to use this codebase to classify medical images are provided in the Jupyter notebooks __tutorials/DICOM_crawler.ipynb__,  __tutorials/headCTclassifier.ipynb__, and __tutorials/walkthrough.ipynb__. We recommend the following order of tutorials to get started:
1. __DICOM_crawler.ipynb__ will demonstrate how to extract pixel data and metadata from a set of DICOM files.
2. __headCTclassifier.ipynb__ will show how to use this codebase to classify a set of medical images stored in a h5 file.
3. __walkthrough.ipynb__ will give an example end-to-end walkthrough on how to use the DICOM crawling and classification pipeline to classify a real dataset.

To enable function calls from the command line, __run_DICOM_crawler.py__ and __run_classifer.py__ are also provided, which perform the same operations as their respective Jupyter notebooks listed above.

## Example usage
We recommend you start with the tutorials listed above, which more thoroughly detail the structure and usage of this codebase. Here, we provide an overview of the steps required to take a set of medical images stored in DICOMs and train a CNN for image classification.

1. Extract the pixel data and metadata from the DICOM files:
```
python run_DICOM_crawler.py ....
```
⋅⋅⋅This command will output metadata_example.csv, containing all metadata from all DICOMs, and pixel_data_example.h5 where all pixel data from the DICOMs are stored. 

2. Compile train.csv, valid.csv, and test.csv files containing the ID and label for each scan, as described in headCTclassifier.ipynb. 

3. Train a classifier for this set of images:
```
python run_classifier.py ....
```
   This command will train a 2d resnet18 to classify the images stored in pixel_data_example.h5 according to the labels in train.csv and valid.csv. It will then evaluate the model according to the labels in test.csv. Files generated from this command include: train_log.txt, which stores all outputs from the training process; predicted_labels.csv, which are the best trained model's predictions on the test set (if provided, otherwise the validation set); learning_curve.png, a simple plot of the training and validation loss curves throughout training; a tensorboard events file, which you can use to view all performance metrics and losses as a function of training the train and validation sets; model_best_train_loss.pth and model_best_val_loss.pth, which contain the stored models that acheived the best loss on the trian and validation set, respectively; and finally all_params.txt, which contains the configuration used this run.
