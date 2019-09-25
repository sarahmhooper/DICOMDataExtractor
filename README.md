# Medical Image Classification and DICOM Information Extraction

We provide a general codebase to perform medical image classification via deep learning using Pytorch. We split the codebase along its two primary functions: data extraction and classification. We have separated the two functions because DICOM information extraction is often needed outside of classification tasks and vice versa.

## Data Extraction from DICOM Files
Medical images are often stored as DICOMs, which can be an inconvenient storage method for machine learning. The metadata stored in each individual DICOM file is inefficient to mine, while grouping all DICOM files from a single scan and stacking the pixel data into a 3d image volume can be a slow process. To address these problems, we provide a DICOM "crawling" script, which provides the following useful functions: 
- Given a set of DICOM files, the DICOM crawling script organizes all metadata from all provided DICOM files into a single CSV, which can be more conviently mined. 
- Given a set of DICOM files, the DICOM crawling script stores all pixel data from all provided DICOMs in an h5 file which can be more efficiently accessed and is more familiar to machine learning practitioners. We also provide functionality to automatically identify DICOM files that are axial slices originating from the same scan, stack the files into a 3d image volume, and save the 3d image in the h5 file.
To learn more about this data extraction code, please view data_extraction/README.md. 

## Medical Image Classification
This codebase was designed to provide a convenient pipeline to take a set of medical images and train a variety of convolutional neural networks (CNN) to classify those images in Pytorch. Features include:

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
Required files, details on the codebase organization, and examples of how to use this codebase to classify medical images are provided in the Jupyter notebookes __tutorials/DICOM_crawler__ and __tutorials/headCTclassifier.ipynb__. 
- __DICOM_crawler__ will demonstrate how to extract pixel data and metadata from a set of DICOM files.
-__headCTclassifier.ipynb__ will show how to use this codebase to classify a set of medical images.

To enable function calls from the command line, __run_DICOM_crawler.py__ and __run_classifer.py__ are provided, which contain the same code as the Jupyter notebooks.

## Example usage
We recommend you start with the two Jupyter notebooks listed above, which more thoroughly detail the structure and usage of this codebase. Here, we provide an overview of the required steps to take a set of medical images stored in DICOMs and train a CNN for image classification.

1. Extract the pixel data and metadata from the DICOM files:
```
python run_DICOM_crawler.py ....
```
2. Manually generate the train.csv, valid.csv, and test. csv files containing the label for each scan, as described in __headCTclassifier.ipynb__.
3. Train a classifier for this set of images:
```
python run_classifier.py ....
```
