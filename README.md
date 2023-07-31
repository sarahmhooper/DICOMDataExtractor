## Data Extraction from DICOM Files
Medical images are often stored as DICOMs, which can be an inconvenient storage method for machine learning. The metadata stored in each individual DICOM file is inefficient to mine, while grouping all DICOM files originating from the same series and stacking the pixel data into a 3d image volume can be a slow process. To address these problems, we provide a DICOM "crawling" script, which provides the following useful functions: 
- Given a set of DICOM files, the DICOM crawling script organizes all metadata from all provided DICOM files into a single CSV, which can be more conviently mined. 
- Given a set of DICOM files, the script stores all pixel data from all provided DICOMs in an HDF5 file which can be more efficiently accessed and is more familiar to machine learning practitioners. 
- If desired, the script will automatically identify DICOM files that originated from the same series, order the slices into a 3d image volume, and save the 3d image in an h5 file. Note: this function assumes the DICOM files represent axial slices - not sagittal or coronal slices - of the same series.

Any comments, suggestions, or additions are welcome!


## Installation 

*Note: Codebase only supports Python 3.* 

To use this code, first clone the repo then install dependencies using ``requirements.txt``; we recommend doing so in a virtual environment. 

## Getting started
__crawler_tutorial.ipynb__ will demonstrate how to extract pixel data and metadata from a set of DICOM files. To enable function calls from the command line, __run_crawler.py__ is also provided to perform the same operations as the Jupyter notebook.

There are a few parameters that you need to set to run the dicom crawling code; these parameters are described in __crawler_tutorial.ipynb__. Alternatively, to view all parameter options for training, run:

```
python run_crawler.py -h
```


## Example usage

#### Example 1. Save all metadata from each DICOM folder without saving any pixel data. 
We can save all of the DICOM metadata from each DICOM file into a CSV without saving any pixel data by setting ``write_pixeldata=False``:
```
python run_crawler.py --dicom_folders "['storage/dicom_folder1','storage/dicom_folder2']" --output_id 'example1' --n_procs 1 --write_pixeldata False --eval_3d_scans False
```
This command will output metadata_example1.csv, containing all metadata from all DICOMs. 

#### Example 2. Save all pixel data from each DICOM as a 2d image and save all DICOM metadata per file
We can save all of the DICOM metadata from each DICOM file into a CSV and save pixel data from each DICOM separately into an h5 file by setting ``write_pixeldata=True`` and ``eval_3d_scans=False``:
```
python run_crawler.py --dicom_folders "['storage/dicom_folder1','storage/dicom_folder2']" --output_id 'example2' --n_procs 1 --write_pixeldata True --eval_3d_scans False
```
This command will output metadata_example2.csv, containing one line of DICOM metadata per DICOM file, and pixel_data_example2.h5 where all 2d images are stored. 

#### Example 3. Save all pixel data from each series of DICOMs into 3d image volumes and save DICOM metadata for each series
We can save the DICOM metadata for each image series into a CSV and save all 3d image volumes in an h5 file by setting ``write_pixeldata=True`` and ``eval_3d_scans=True``:
```
python run_crawler.py --dicom_folders "['storage/dicom_folder1','storage/dicom_folder2']" --output_id 'example3' --n_procs 20 --write_pixeldata True --eval_3d_scans True
```
This command will output metadata_example3.csv, containing one line of DICOM metadata per series (3d image volume), and pixel_data_example3.h5 where all image volumes are stored. When grouping DICOMs to store 3d scans, this code will check all DICOMs in a given folder for unique Series Instance UIDs. If multiple Series Instance UIDs are present in the folder, indicating more than one scan present in that folder, then DICOMs with the same UID will be grouped and stored together. 


## Additional notes

 - The code may generate ``unreadable_files.csv``, which contains a row for each scan that did not successfully have metadata and/or pixel data extracted; if a Scan ID appears in unreadable_files.csv, it should not appear in either pixel_data.h5 or metadata.csv. A brief reason indicating why the read failed is provided in each row.

 - If output files pixel_data.h5 and metadata.csv already exist, running this code will append onto the existing files with new scans; any scans already stored will be ignored. This is useful if you acquire new data you want to add to your dataset, or if your dicom_crawl() call is stopped sometime during execution.

 - Sometimes, there will be duplicate tags in the DICOM header. If this is the case, all values associated with that tag will be stored in the same cell in metadata.csv. For example, if the tag "Example tag (0000, 0000)" appears twice in the DICOM header, once with the value "1" and once with the value "2", then the stored metadata for "Example tag (0000, 0000)" will appear as "DUPLICATE TAGS IN DICOM:1/2".

 - If the header row in metadata.csv contains two phrases connected by an underscore (e.g., "Referenced Image Sequence_Group Length (0008, 0000)") this indicates that the second tag ("Group Length") was nested inside of the first tag ("Referenced Image Sequence").

