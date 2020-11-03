import os
import importlib

def get_data(list_of_data):

    DOWNLOAD_DIR = os.path.join(".", "data", "raw")
    os.makedirs(DOWNLOAD_DIR, exist_ok = True)

    # If we have passed an empty list to the downloader, then we will download every available dataset
    if len(list_of_data) < 1:
        downloaders_dir = os.path.join(".", "data", "downloaders")
        downloaders_file_list = os.listdir(downloaders_dir)
        list_of_data = [file_name[:-3] for file_name in downloaders_file_list if file_name[-3:] == ".py"]

    def get_single_dataset(dataset_name):
        downloader_module = importlib.import_module(f"data.downloaders.{dataset_name}")
        print(f"Now downloading {dataset_name}")
        downloader_module.Downloader(DOWNLOAD_DIR, dataset_name).save_data()

    for dataset in list_of_data:
        get_single_dataset(dataset)
