from crawler_utils import dicom_crawl, parse_arg
import glob
import os

def run_dicom_crawl():
    
    # Get arguments
    parser = parse_arg(args=[])
    config = parser.parse_args()
    
    # Crawl dicoms
    dicom_crawl(config.dicom_folders, 
                config.storage_folder, 
                config.output_id, 
                config.n_procs, 
                config.write_pixeldata, 
                config.eval_3d_scans, 
                config.par_over_folder)
    
    print("Finished running DICOM crawling code.")


if __name__ == "__main__":
    run_dicom_crawl()
   