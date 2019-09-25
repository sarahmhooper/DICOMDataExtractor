# Medical Image Classification and DICOM Information Extraction

We provide a general codebase to perform medical image classification via deep learning using Pytorch. We split the codebase into two primary functions: data extraction and classification. 

## Data Extraction from DICOM Files
Medical images are often stored as DICOMs, which can be an inconvenient storage method for machine learning. The metadata stored in each individual DICOM file is inefficient to mine, while grouping all DICOM files from a single scan and stacking the pixel data into a 3d image volume can be a slow process. To address these problems, we provide a DICOM "crawling" script, which provides the following useful functions: 
- Given a set of DICOM files, the DICOM crawling script will organizes all metadata from all provided DICOM files into a single CSV, which can be more conviently mined. 
- Given a set of DICOM files, the DICOM crawling script will store all pixel data from all provided DICOMs in an h5 file which can be more efficiently read from during training and more familiar to machine learning practitioners. We also provide functionality to automatically identify DICOM files that originate from the same scan, stack the files into a 3d image volume, and save the 3d image in the h5 file.
To learn more about this data extraction code, please view data_extraction/README.md

## Medical Image Classification
This codebase was designed to provide a convenient pipeline to take a set of medical images and train a variety of convolutional neural networks (CNN) to classify those images. Features include:

- Automated data loading, training, and model evaluation
- Implementation of many common CNNs
- Support for 2d, 2.5d, and 3d medical image classification
- Support for weak supervision using text reports
- Training monitoring via tensorboardX


This project is ongoing, and we'll continue to add to and adjust the codebase over time. Any comments, suggestions, or additions are welcome!
