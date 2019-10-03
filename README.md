## Data Extraction from DICOM Files
Medical images are often stored as DICOMs, which can be an inconvenient storage method for machine learning. The metadata stored in each individual DICOM file is inefficient to mine, while grouping all DICOM files from a single scan and stacking the pixel data into a 3d image volume can be a slow process. To address these problems, we provide a DICOM "crawling" script, which provides the following useful functions: 
- Given a set of DICOM files, the DICOM crawling script organizes all metadata from all provided DICOM files into a single CSV, which can be more conviently mined. 
- Given a set of DICOM files, the script stores all pixel data from all provided DICOMs in an h5 file which can be more efficiently accessed and is more familiar to machine learning practitioners. 
- If desired, the script will automatically identify DICOM files that originated from the same scan, order the slices into a 3d image volume, and save the 3d image in an h5 file.

*Note: to use the resulting files to train a classifier in Pytorch, see our MedImgClassifier repository.*

This project is ongoing, and we'll continue to add to and adjust the codebase over time. Any comments, suggestions, or additions are welcome!


## Installation 

*Note: Codebase only supports Python 3.*
*Note: If using DICOM crawling code with the classification pipeline, you do not need to repeat installation here.* 

1. Create a new virtual environment (or activate an existing environment), within which we'll install the codebase and its dependencies:
```
pip install virtualenv
virtualenv -p python3.6 DICOMCrawlEnv
```

2. Activate the desired virtual environment:
```
source DICOMCrawlEnv/bin/activate
```

3. Clone the repository:
```
git clone https://github.com/sarahmhooper/DICOMDataExtractor.git
```

4. Install the required Python dependencies: 
```
pip install -r requirements.txt
```

## Getting started
__crawler_tutorial.ipynb__ will demonstrate how to extract pixel data and metadata from a set of DICOM files. To enable function calls from the command line, __run_crawler.py__ is also provided to perform the same operations as the Jupyter notebook.

## Example usage
To view all parameter options for training, run:

```
python run_crawler.py -h
```
### Example 1. Save all metadata friom each DICOM folder without saving any pixel data. 

### Example 2. Save the DICOM metadata from every DICOM file 
Given a set of folders in the directory *storage/dicom_folders*, we can save all of the DICOM metadata from each individual DICOM file into a CSV (without saving any pixel data) by running:
```
python run_DICOM_crawler.py ....
```
This command will output metadata_example2.csv, containing all metadata from all DICOMs. 

### Example 3. Save all pixel data into 3d image volumes and save DICOM metadata per scan
Given a set of folders in the directory *storage/dicom_folders*, we can save all of the DICOM metadata into a CSV and save all 3d scans in a h5 file by running:
```
python run_DICOM_crawler.py ....
```
This command will output metadata_example1.csv, containing one line of DICOM metadata per 3d scan, and pixel_data_example1.h5 where all 3d images are stored. 
