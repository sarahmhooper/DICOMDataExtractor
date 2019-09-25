# Medical Image Classification

**v0.1.0**

We provide a general codebase to perform medical image classification via deep learning using Pytorch. This codebase was designed to provide a convenient pipeline to take a set of medical images, originally (but not necessarily) stored as DICOM files, and train a variety of convolutional neural networks (CNN) to classify those images. Features include:
- Tools to extract all pixeldata and metadata from a set of DICOM files
- Implementation of many common CNNs
- Data loading, training, and model evaluation
- Support for 2d, 2.5d, and 3d medical image classification
- Support for weak supervision using text reports
- Training monitoring via tensorboardX

This project is ongoing, and we'll continue to add to and adjust the codebase over time. Any comments, suggestions, or additions are welcome! 

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
Required files, details on the codebase organization, and an example of how to use this codebase to classify medical images are provided in two Jupyter notebookes: __DICOM_crawler.ipynb__ and __headCTclassifier.ipynb__. 
- To learn how to extract all pixel data and metadata from a set of DICOM files, see __DICOM_crawler.ipynb__
- To learn how to use this codebase to classify a set of images, see __headCTclassifier.ipynb__

To enable function calls from the command line, two .py files are provided which contain the same code as the Jupyter notebooks: __run_DICOM_crawler.py__ and __run_classifier.py__.

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
