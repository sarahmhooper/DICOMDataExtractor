# Medical Image Classification Codebase

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

## Running 
organize data
can do jupyter notebook or python run.py

## Sample usage
Required files, details on the codebase organization, and an example of how to use this codebase to classify medical images are provided in two Jupyter notebookes: __DICOM_crawler.ipynb__ and __headCTclassifier.ipynb__. 
- To learn how to extract all pixel data and metadata from a set of DICOM files, see __DICOM_crawler.ipynb__
- To learn how to use this codebase to classify a set of images, see __headCTclassifier.ipynb__
