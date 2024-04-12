import requests
from io import BytesIO
from zipfile import ZipFile
import os
import pandas as pd

def download_zip_and_extract(url, destination_folder):
    # Make a request to the URL
    response = requests.get(url)

    if response.status_code == 200:
        # Create a ZipFile from the response content
        with ZipFile(BytesIO(response.content)) as zipped_file:
            # Extract all contents to the destination folder
            zipped_file.extractall(destination_folder)

def download_tsv_from_folder(folder_path, tsv_filename):
    # List all files in the folder
    # files = os.listdir(folder_path)

    # Find the TSV file in the list
    # tsv_file = next((file for file in files if file.endswith('.tsv')), None)

    if tsv_filename:
        # Read the TSV file using pandas
        tsv_data = pd.read_csv(os.path.join(folder_path, tsv_filename), sep='\t')
        return tsv_data
    else:
        print("No TSV file found in the folder.")

# Example Usage:
zip_url = "https://www.sec.gov/dera/data/form-13f/2023q3_form13f.zip"
destination_folder = "/Users/christianruiz/Desktop/github/comp-642/final_project/hist_data/"

# Download and extract the zip file
download_zip_and_extract(zip_url, destination_folder)

# Download TSV from the extracted folder
tsv_filename = "INFOTABLE.tsv"
tsv_data = download_tsv_from_folder(destination_folder, tsv_filename)

# Now you can work with tsv_data
print(tsv_data)