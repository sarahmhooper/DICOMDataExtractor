from __future__ import print_function
import os
import re
import magic
import pydicom
import h5py
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool, Lock
from functools import partial
from tqdm import *
from collections import defaultdict

def dataset_tags_and_values(dataset, tags=[], values=[], starting_seq='', loop=0):
    '''
    This function finds all metadata (i.e., tags and corresponding values) contained in a pydicom dataset, excluding the pixel data.
    :param dataset: a pydicom dataset; contains all information from a DICOM file
    :param tags: List of tags in the form (name, numerical tag) from the DICOM header - don't alter, will be filled out by this function
    :param values: List of values from the DICOM header, corresponding to tags - don't alter, will be filled out by this function
    :param starting_seq: a string that will be prepended to the tags name - don't alter, will be filled by this function
    :param loop: an int tracking how many nested sequences we are traversing - don't alter, will be tracked by this function
    :return tags: the completed list of tags in the DICOM dataset
    :return values: the completed list of values in the DICOM dataset
    '''
    for data_element in dataset:
        if data_element.VR == "SQ":  # Determine if the data element is a sequence (if so, this element's value itself contains a pydicom dataset)
            loop += 1  # Track that we have entered a sequence
            starting_seq = starting_seq + data_element.name + '_'
            for seq_dataset in data_element.value:  # Retrieve each dataset contained in the sequence
                tags, values = dataset_tags_and_values(seq_dataset, tags, values, starting_seq,loop)  # Find all tags and values in the dataset contained in the sequence
            loop -= 1  # Track that we have finished traversing a sequence
            # Adjust name of tag to only include sequences we are still traversing
            new_starting_seq = ''
            for entry in starting_seq.split('_')[:loop]: new_starting_seq += entry + '_'
            starting_seq = new_starting_seq
        else:  # If the data element is not a sequence, store the tag name and value for all metadata that is not the pixel data
            if data_element.name not in ["Pixel Data"]:
                # If this tag has not been previously stored, store value
                if starting_seq + data_element.name + ' ' + str(data_element.tag) not in tags:
                    tags.append(starting_seq + data_element.name + ' ' + str(data_element.tag))
                    values.append(str(data_element.value))
                else: # If this tag has been previously stored, append value
                    tag = starting_seq + data_element.name + ' ' + str(data_element.tag)
                    if values[tags.index(tag)][0:24] == 'DUPLICATE TAGS IN DICOM:':
                        values[tags.index(tag)] = values[tags.index(tag)]+'/'+str(data_element.value)
                    else:
                        values[tags.index(tag)] = 'DUPLICATE TAGS IN DICOM:'+values[tags.index(tag)]+'/'+str(data_element.value)
    return tags, values

def order_image_num(dicoms):
    '''
    This function orders dicoms based on the image number found in each dicom's metadata.
    :param dicoms: A list containing paths to all dicoms for this scan
    :return slice_order: List of ordered indices.
    '''
    image_numbers = [pydicom.dcmread(dicom).InstanceNumber for dicom in dicoms]
    slice_order = sorted(range(len(dicoms)), key=lambda i: image_numbers[i])
    return slice_order

def order_z_pos(dicoms):
    '''
    This function orders dicoms based on the z position found in each dicom's metadata.
    :param dicoms: A list containing paths to all dicoms for this scan
    :return slice_order: List of ordered indices.
    '''
    z_pos = []
    for dicom in dicoms:
        z_pos += [float(pydicom.dcmread(dicom).ImagePositionPatient[-1])]
    slice_order = sorted(range(len(dicoms)), key=lambda i: z_pos[i])  # Sorts z position in ascending order
    return slice_order

def order_fn_num(dicoms):
    '''
    This function orders dicoms based on their filenames.
    :param dicoms: A list containing paths to all dicoms for this scan
    :return slice_order: List of ordered indices.
    '''
    slice_numbers = ["".join(re.findall('\d+',os.path.basename(dicoms[ind]))) for ind in range(len(dicoms))] # Obtain the numbers in all the DICOM filenames
    slice_order = sorted(range(len(dicoms)),key=lambda i: slice_numbers[i])  # Order the numbers from smallest to largest
    return slice_order

def get_pixel_data(dicoms, save_3d_scans):
    '''
    This function creates a numpy array of pixel values obtained from the filenames in dicoms.
    :param dicoms: A list containing paths to all dicoms for this scan
    :param save_3d_scans: bool indicating if 2d or 3d images should be saved 
    :return successful_access: True if the pixel data was successfully accessed and a False else
    :return pixeldata: Numpy array containing all pixel data
    '''
    try:
        # Determine slice order based off of metadata image number, image z position, then number in filename
        for fn in [order_image_num, order_z_pos, order_fn_num]:
            try:
                slice_order = fn(dicoms)
                break
            except: continue
        # Create 3d array with all pixel data
        first_scan = pydicom.dcmread(dicoms[0]).pixel_array
        if len(dicoms)>1:
            pixeldata = np.empty((first_scan.shape[0], first_scan.shape[1], len(dicoms)), dtype=first_scan.dtype)
            for slice_ind, fn_ind in enumerate(slice_order):
                pixeldata[:, :, slice_ind] = pydicom.dcmread(dicoms[fn_ind]).pixel_array
        else: 
            pixeldata = first_scan
            if save_3d_scans: pixeldata = np.expand_dims(pixeldata,2)
        pixeldata = np.expand_dims(pixeldata,0) # Give pixeldata channel dimension = 1 at axis 0
        successful_access = True
    except: # If there's an error, catch it and record unsuccessful attempt
        successful_access = False
        pixeldata = []
    return successful_access, pixeldata

def extract_data(dicoms, scan_id, write_pixeldata, save_3d_scans):
    '''
    This function is used to extract pixel and header data from a list of dicoms.
    :param dicoms: List of filepaths to all DICOMs in a scan.
    :param scan_id: Unique identifier, constructured from filepath and Series Instance UID (if available)
    :param write_pixeldata: Binary value indicating if metadata h5 should be written.
    :param save_3d_scans: bool indicating if 2d or 3d images should be saved 
    :return successful_access: Boolean - true if the pixel/meta data was successfully obtained and false else
    :return reason: String containing brief description of error that caused unsuccessful access
    :return tags: List of tags in the form (name, numerical tag) from the dicom header
    :return values: List of values from the dicom header, corresponding to tags
    :return pixeldata: Numpy array containing all pixel data from dicoms
    '''
    successful_access = False
    reason = 'Unidentified failure'
    if len(dicoms) > 0:
        try:
            # if write_pixeldata=False, do not load in all pixeldata to check successful_access
            if write_pixeldata:
                successful_access, pixeldata = get_pixel_data(dicoms, save_3d_scans)  # Returns nparray of pixeldata from dicoms
            else:
                successful_access = True
                pixeldata = []
            if successful_access:
                # Find all metadata tags and corresponding values from a DICOM; NOTE: this code only stores the metadata for ONE DICOM file in dicoms (i.e. it assumes that all needed dicom header information is the same for all slices)
                tags, values = dataset_tags_and_values(pydicom.dcmread(dicoms[0]),tags=['Scan ID','No. DICOMs in scan'], values=[scan_id,len(dicoms)])  # Initialize metadata tags and values
            else: reason = "Failure in get_pixel_data"
        except:
            successful_access = False
            reason = "Failure in dataset_tags_and_values"
    else: reason = "Number of dicoms in folder = 0"
    if not successful_access:  # If any pixel/meta data access fails, store only basic information about dicoms and ignore pixeldata/other metadata
        print("\n\tUnsuccessful DICOM read for scan_id:", scan_id+";",reason)
        tags = ['Scan ID', 'No. DICOMs in scan']
        values = [scan_id, len(dicoms)]
        pixeldata = []
    return successful_access, reason, tags, values, pixeldata

def crawl_uid(uid_dicoms, folder, existing_scan_ids, pixeldata_storage_fn, metadata_storage_fn, unreadable_storage_fn, write_pixeldata, save_3d_scans):
    '''
    This function extracts pixel and header data from a folder of DICOM images and writes that information to .h5 and .csv files.
    :param uid_dicoms: A tuple containing (a string containing the unique identifier for this scan, a list of dicom filenames included in this scan)
    :param folder: A string containing the path to the folder containing dicom files.
    :param existing_scan_ids: A dict containing scan IDs that are already stored from a previous run.
    :param pixeldata_storage_fn: A string containing the filename to store the pixel data.
    :param metadata_storage_fn: A string containing the filename to store the dicom metadata.
    :param unreadable_storage_fn: A string containing the filename to store the DICOM folders that fail to read.
    :param write_pixeldata: Binary value indicating if metadata h5 should be written.
    :param save_3d_scans: bool indicating if 2d or 3d images should be saved 
    :return nothing, files are written to within this function
    '''
    # Create scan_id, used to uniquely identify each set of DICOMs in all output files, and gather all DICOMs with correct UID
    scan_id = folder.replace("/","_").replace("\\","_").replace("//","_") + "_" + uid_dicoms[0]
    
    # Only process scans that were not stored by a previous run
    if scan_id not in existing_scan_ids:
       
        # Extract all data from set of DICOMs
        successful_access, reason, tags, values, pixeldata = extract_data(uid_dicoms[1], scan_id, write_pixeldata, save_3d_scans)

        # Make sure files are only being written to by this processor
        if save_3d_scans: lock.acquire()
        
        # If there was a successful read of the DICOMs, store all data.
        if successful_access:
            try:
                # Open h5 file to store all pixel data, if desired
                if write_pixeldata:
                    with h5py.File(pixeldata_storage_fn, mode="a", libver='latest') as pixeldata_storage_h5:
                        pixeldata_storage_h5.create_dataset(scan_id, shape=pixeldata.shape, data=pixeldata)  # Save matrix in h5 file with scan_id as the title; options: specify dtype or compression for more efficient storage (may increase read time)
                if not os.path.exists(metadata_storage_fn):
                    # Write metadata for first scan
                    metadata = pd.DataFrame(columns=tags)
                    metadata.loc[0] = values
                    metadata.to_csv(metadata_storage_fn, index=False)
                else:
                    new_metadata = pd.DataFrame([dict(zip(tags,values))])
                    existing_headers = pd.read_csv(metadata_storage_fn, nrows=1).columns
                    if set(tags).issubset(set(existing_headers)):
                        new_metadata.reindex(columns=existing_headers).to_csv(metadata_storage_fn, index=False, header=False, mode="a") 
                    else:
                        new_metadata = pd.read_csv(metadata_storage_fn).append(new_metadata, ignore_index=True)                
                        new_metadata.to_csv(metadata_storage_fn, index=False,  mode='w')
            except:
                successful_access = False
                reason = 'CSV or pixel data write failure' 

        # If there was a failed read of the DICOMs, store info in unreadable_metadata_fn
        if not successful_access:
            if not os.path.exists(unreadable_storage_fn):
                # Store first unreadable scan
                unreadable = pd.DataFrame(columns = ['Reason for failure']+tags)
                unreadable.loc[0] = [reason]+values
                unreadable.to_csv(unreadable_storage_fn, index=False)
            else:
                # Append additional unreadable scans
                old_unreadable = pd.read_csv(unreadable_storage_fn)
                new_unreadable = old_unreadable.append(dict(zip(['Reason for failure']+tags,[reason]+values )),ignore_index=True)
                new_unreadable.to_csv(unreadable_storage_fn, index=False)
        if save_3d_scans: lock.release()  # Release to other processors
    return

def crawl_folder(folder, existing_scan_ids, pixeldata_storage_fn, metadata_storage_fn, unreadable_storage_fn, write_pixeldata, save_3d_scans, par_over_folder, n_procs):
    '''
    This function extracts pixel and header data from a folder of DICOM images and writes that information to .h5 and .csv files.
    :param folder: A string containing the path to the folder containing dicom files.
    :param existing_scan_ids: A dict containing scan IDs that are already stored from a previous run.
    :param pixeldata_storage_fn: A string containing the filename to store the pixel data.
    :param metadata_storage_fn: A string containing the filename to store the dicom metadata.
    :param unreadable_storage_fn: A string containing the filename to store the DICOM folders that fail to read.
    :param write_pixeldata: Binary value indicating if metadata h5 should be written.
    :param save_3d_scans: bool indicating if 2d or 3d images should be saved 
    :param par_over_folder: Binary value indicating if parallelization occurs over the folder or uid w/in a folder
    :return nothing, files are written to within this function
    '''
    # Collect all DICOMs in folder
    if not par_over_folder: 
        print('Ensuring all files in folder are DICOMs, this may take seconds to tens of minutes depending on the number of DICOMs in the folder.')
    all_dicoms = []
    for f in os.listdir(folder):
        if os.path.isfile(os.path.join(folder,f)):
            try:
                if "dicom" in str.lower(magic.from_file(os.path.join(folder,f))): all_dicoms += [os.path.join(folder,f)]
            except: 
                print("DICOM file check failed on "+os.path.join(folder,f)+", file excluded from crawl.")

    # To store 3d scans, store series UIDs to collect all slices in the same series and extract data for that UID
    if save_3d_scans: 
        # Collect all unique UIDs in the folder (each UID will be stored as a different dataset)
        if not par_over_folder: 
            print('Collecting and organizing all DICOM UIDs into 3d scans, this may take seconds to a few hours depending on the number of DICOMs in the folder.')
        try:
            uid_fns = defaultdict(list) 
            for d_ind, dicom in enumerate(all_dicoms):
                uid_fns[pydicom.dcmread(dicom).SeriesInstanceUID] += [dicom]
            if len(uid_fns)>1:
                print("\nThere are",len(uid_fns),"scans in this folder.")
        # If some DICOMs do not have a UID, all DICOMs in the folder will be stored as one dataset
        except: 
            uid_fns = {'nouid':all_dicoms}
                
        # If not parallelizing within a folder and saving 3d scans, crawl each uid in the folder
        if par_over_folder:
            # Loop over each set of DICOMs with the same UID (or all DICOMs if UIDs not in metadata), extract data, and write to output files
            for uid_fn in uid_fns.items(): crawl_uid(uid_fn, folder, existing_scan_ids, pixeldata_storage_fn, metadata_storage_fn, unreadable_storage_fn, write_pixeldata, save_3d_scans)

        # If parallelizing within a folder and saving 3d scans, do so here
        else: 
            l = Lock()
            pool = Pool(processes=n_procs, initializer=init, initargs=(l,))
            # Show progress bar with tqdm
            with tqdm(total=len(uid_fns)) as pbar:
                # Run DICOM crawling code; distribute folders in dicom_folders to different processors
                for i, uid_fn in tqdm(enumerate(pool.imap_unordered(partial(crawl_uid,
                                                                       folder=folder,
                                                                       existing_scan_ids=existing_scan_ids,
                                                                       pixeldata_storage_fn=pixeldata_storage_fn,
                                                                       metadata_storage_fn=metadata_storage_fn,
                                                                       unreadable_storage_fn=unreadable_storage_fn,
                                                                       write_pixeldata=write_pixeldata,
                                                                       save_3d_scans=save_3d_scans),
                                                               uid_fns.items()))):
                    pbar.update()
            pool.close()
            pool.join()

    # If not saving 3d scans, show progress bar for every file read/written
    if not save_3d_scans:
        for dicom_ind in tqdm(range(len(all_dicoms))):
            try: uid = pydicom.dcmread(all_dicoms[dicom_ind]).SOPInstanceUID
            except: uid = os.path.splitext(os.path.split(all_dicoms[dicom_ind])[1])[0]
            crawl_uid((uid,[all_dicoms[dicom_ind]]), folder, existing_scan_ids, pixeldata_storage_fn, metadata_storage_fn, unreadable_storage_fn, write_pixeldata, save_3d_scans)
        # Make sure that the 'no dicom' error message is saved, which is done within crawl_uid
        if len(all_dicoms) == 0: 
            crawl_uid(('nouid',[]), folder, existing_scan_ids, pixeldata_storage_fn, metadata_storage_fn, unreadable_storage_fn, write_pixeldata, save_3d_scans)   
    
    return

def dicom_crawl(dicom_folders, storage_folder, output_id, n_procs, write_pixeldata, save_3d_scans, par_over_folder):
    '''
    This function checks user-input variables, creates filenames, and manages the multiprocessor crawling.
    :param dicom_folders: list of folder paths that contain DICOMs
    :param storage_folder: string containing path to folder where outputs will be stored
    :param output_id: string or None that will be appended to output files
    :param n_procs: number of processors to use
    :param write_pixeldata: bool indicating if pixel data should be written this run
    :param save_3d_scans: bool indicating if 2d or 3d images should be saved
    :param par_over_folder: bool indicating if parallelization occurs over the folder or uid w/in a folder
    :return:
    '''

    # Create filenames for all outputs
    identifier = "" if output_id is None else "_" + output_id  # Identifier to modify output filenames
    metadata_storage_fn = os.path.join(storage_folder,"metadata" + identifier + ".csv")  # Filename (.csv) to store all metadata from DICOM headers
    pixeldata_storage_fn = os.path.join(storage_folder,"pixel_data" + identifier + ".h5")  # Filename (.h5) to store all pixel data
    unreadable_storage_fn = os.path.join(storage_folder,"unreadable_files" + identifier + ".csv")  # Filenames (.csv) to store metadata of files that did not successfully have pixel and/or metadata read

    # Make sure provided dicom_folders is (a) a list and (b) contains folders
    if not isinstance(dicom_folders, list):
        raise ValueError("\nUser-input dicom_folders is not a list.")
    for folder_ind in range(len(dicom_folders) - 1, -1, -1):
        if not os.path.isdir(dicom_folders[folder_ind]):
            print("\nWarning: user input a folder (below) in dicom_folders that is not a directory; item will be removed from dicom_folders.")
            print("\t Removing: ", dicom_folders[folder_ind])
            del dicom_folders[folder_ind]

    # Make sure number of entries in stored metadata and pixeldata match; otherwise, throw error and alert user.
    if write_pixeldata:
        if os.path.exists(metadata_storage_fn) != os.path.exists(pixeldata_storage_fn):
            raise ValueError("\Either pixel data file OR metadata file exists, but not both \nPlease specify a new output_id or delete existing output files and try again.")
        elif os.path.exists(metadata_storage_fn) and os.path.exists(pixeldata_storage_fn):
            with h5py.File(pixeldata_storage_fn, mode="r") as h5file:
                if set(h5file.keys()) != set(pd.read_csv(metadata_storage_fn)["Scan ID"]):
                    raise ValueError("\Existing pixeldata H5 and metadata CSV do not have matching sets of scan IDs \nPlease specify a new output_id or delete existing output files and try again.")

    # Check to see if output files already exist; if so, this code will skip scans already stored and append new scans to existing files
    existing_scan_ids = []
    if os.path.exists(metadata_storage_fn):
        print("\nMetadata CSV already exists. This run will append to the existing file.")
        print("\tExisting metadata filepath is:", metadata_storage_fn)
        try:
            metadata = pd.read_csv(metadata_storage_fn)
            existing_scan_ids += list(metadata["Scan ID"].values)
        except:
            pass
    if write_pixeldata and os.path.exists(pixeldata_storage_fn):
        print("\nPixel data H5 already exists. This run will append to the existing file.")
        print("\tExisting pixel data filepath is:", pixeldata_storage_fn)
    if os.path.exists(unreadable_storage_fn):
        print("\nUnreadable file CSV already exists. This run will append to the existing file.")
        print("\tExisting unreadable files filepath is:", unreadable_storage_fn)
        try:
            unreadable = pd.read_csv(unreadable_storage_fn)
            existing_scan_ids += list(unreadable["Scan ID"].values)
        except:
            pass
    print("\nNumber of previously stored scans: ", len(existing_scan_ids))
    existing_scan_ids = dict.fromkeys(existing_scan_ids, True)
    
    # Set up pixel data storage h5, if needed
    print("Starting DICOM crawling...\n")
    # Set up multiprocessing over folder
    if par_over_folder and save_3d_scans:
        l = Lock()
        pool = Pool(processes=n_procs, initializer=init, initargs=(l,))
        # Show progress bar with tqdm
        with tqdm(total=len(dicom_folders)) as pbar:
            # Run DICOM crawling code; distribute folders in dicom_folders to different processors
            for i, _ in tqdm(enumerate(pool.imap_unordered(partial(crawl_folder,
                                                                   existing_scan_ids=existing_scan_ids,
                                                                   pixeldata_storage_fn=pixeldata_storage_fn,
                                                                   metadata_storage_fn=metadata_storage_fn,
                                                                   unreadable_storage_fn=unreadable_storage_fn,
                                                                   write_pixeldata=write_pixeldata,
                                                                   save_3d_scans=save_3d_scans,
                                                                   par_over_folder=par_over_folder,
                                                                   n_procs=n_procs),
                                                           dicom_folders))):
                pbar.update()
        pool.close()
        pool.join()
    else:
        for f_ind,folder in enumerate(dicom_folders):
            print('Working on folder',f_ind,'of',len(dicom_folders),', named:',folder)
            crawl_folder(folder,
                         existing_scan_ids,
                         pixeldata_storage_fn,
                         metadata_storage_fn,
                         unreadable_storage_fn,
                         write_pixeldata,
                         save_3d_scans,
                         par_over_folder,
                         n_procs)

def parse_arg():
    """
    Used to parse arguments from the command line
    """
    parser = argparse.ArgumentParser(
            "CrawlerConfiguration",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--dicom_folders",
        type=obj_to_str_list,
        required=True,
        help="""A list of string directories, each of which should contain DICOMs; formatting from command line should follow: "['/path/to/folder/1', '/path/to/folder/2']" """,
    )
    
    parser.add_argument(
        "--storage_folder", 
        type=str, 
        default=os.getcwd(), 
        help="String directory where outputs are to be stored",
    )

    parser.add_argument(
        "--output_id",
        type=str,
        default=None,
        help="A unique string identifier for output filenames; can be None. ex: output_id='study1' results in 'pixel_data_study1.h5', output_id=None results in 'pixel_data.h5'",
    )
    
    parser.add_argument(
        "--n_procs",
        type=int,
        default=1,
        help="Number of processors",
    )

    parser.add_argument(
        "--write_pixeldata",
        type=confirm_bool,
        default=False,
        choices=[True,False],
        help="Boolean indicator for whether to write pixel data to an h5 file",
    )
    
    parser.add_argument(
        "--eval_3d_scans",
        type=confirm_bool,
        default=False,
        choices=[True,False],
        help="Boolean indicator for whether to evaluate 3d scans or 2d images. If True, dicom_crawl() will find all scans with the same series instance UID and stack them in order into a 3d image, then save the 3d stack in the h5 and save one line of metadata/3d scan. If False, dicom_crawl() will save each individual DICOM file's pixel data as a 2d image in the h5 file and save each DICOM file's metadata in the CSV.",
    )
        
    parser.add_argument(
        "--par_over_folder",
        type=confirm_bool,
        default=False,
        choices=[True,False],
        help="Choose whether to parallelize over the folders by setting par_over_folder to True, or over the scans within a folder by setting par_over_folder to False. If you have many folders in dicom_folders, each with O(1) scan, set to True. If you have many scans per folder, set to False. Note this parallelization is only used when evaluating 3d data.",
    )
    
    return parser

def confirm_bool(inputstr):
    """
    Ensure user input (inputstr) is interpreted as a bool
    """
    if str(inputstr.strip()).lower() in ['yes','true','1','t','y']:
        return True
    elif str(inputstr.strip()).lower() in ['no','false','n','0','f']:
        return False
    else:
        raise ValueError('Please ensure boolean variable input.')
        
def obj_to_str_list(inputstr):
    """
    Convert user input (inputstr) to a list of strings
    """
    all_items = inputstr.strip().strip("[]")
    all_items = [str(i.strip().strip("''").strip('""').strip()) for i in all_items.split(",")]
    return all_items

def init(l):
    '''
    This function is used by multiprocessing to lock before writing to files
    '''
    global lock
    lock = l
