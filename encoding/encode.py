import os
import importlib
import pandas as pd
import numpy as np

def get_encodings(list_of_encoders):

    ENCODING_DIR = os.path.join(".", "data", "encoding")
    os.makedirs(ENCODING_DIR, exist_ok = True)

    # If we have passed an empty list to the downloader, then we will download every available dataset
    if len(list_of_encoders) < 1:
        encoders_dir = os.path.join(".", "encoding", "encoders")
        encoders_file_list = os.listdir(encoders_dir)
        list_of_encoders = [file_name[:-3] for file_name in encoders_file_list if file_name[-3:] == ".py"]

        # Make sure that all the encoders that rely on concatenation come at the end
        list_of_encoders_non_concat = [enc_name for enc_name in list_of_encoders if "concat" not in enc_name]
        list_of_encoders_concat = [enc_name for enc_name in list_of_encoders if "concat" in enc_name]
        list_of_encoders = list_of_encoders_non_concat + list_of_encoders_concat

    # Get the raw data to encode in a {"dataset_name": dataset_df}
    def get_dataset_dict():
        data_folder_dir = os.path.join(".", "data", "raw")
        data_file_list = os.listdir(data_folder_dir)
        list_of_datasets = [file_name for file_name in data_file_list if file_name[-4:] == ".csv"]

        dataset_dict = {}
        for dataset in list_of_datasets:
            dataset_dir = os.path.join(data_folder_dir, dataset)
            data_df = pd.read_csv(dataset_dir, index_col=0)
            dataset_dict[dataset[:-4]] = data_df
        return dataset_dict

    # Save the encoding with a column for labels in the ./data/encoding/dataset_name/encoding_name.csv path
    def save_encoding(dataset_name, encoder_name, encoding, labels):
        dataset_encoding_file_path = os.path.join(ENCODING_DIR, f"{dataset_name}")
        os.makedirs(dataset_encoding_file_path, exist_ok = True)
        encoding_file_path = os.path.join(dataset_encoding_file_path, f"{encoder_name}.npy")

        df = pd.DataFrame(encoding)
        df["labels"] = labels

        with open(encoding_file_path, "wb") as f:
            np.save(f, df.values)

    # Get one encoding type for all datasets currently downloaded
    def get_single_encoding_type(encoder_name):
        encoder_module = importlib.import_module(f"encoding.encoders.{encoder_name}")
        print(f"Now loading {encoder_name}")
        encoder = encoder_module.Encoder()

        dataset_dict = get_dataset_dict()

        for dataset_name, dataset_df in dataset_dict.items():
            print(f"Now encoding {dataset_name} using {encoder_name}")
            encoding = encoder.encode(list(dataset_df.text), dataset_name=dataset_name)
            save_encoding(dataset_name, encoder_name, encoding, list(dataset_df.labels))

    for encoder_name in list_of_encoders:
        get_single_encoding_type(encoder_name)
