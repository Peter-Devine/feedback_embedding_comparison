import os
import pandas as pd

class DownloadUtilBase:
    def __init__(self, download_dir, dataset_name):
        self.dataset_name = dataset_name
        self.download_dir = download_dir

    def download(self):
        # The download function should output a dictionary like {"subset_name": df}
        # If the dataset only has one subset (I.e. the dataset should not be split up)
        # then the downloader should simply output a dict of one key-value pair
        # which looks like {"all": df}
        raise NotImplementedError(f"No self.download() function has been initialised for {self.get_app_name()}")

    def save_data(self):
        MIN_DATA_SIZE = 10

        downloads_dfs_dict = self.download()

        for subset_name, df in downloads_dfs_dict.items():

            # Each dataset needs to have a "text" column and a "label" column
            required_cols = ["text", "label"]
            assert all([required_col in df.columns for required_col in required_cols]), f"Missing one of {required_cols} in df columns ({df.columns})"

            # In case any rows of text are repeated (I.e. have multiple labels) we add them all together here into one list
            grouped_df = df.groupby("text").label.apply(list)
            df = pd.DataFrame({"text": grouped_df.index, "labels": grouped_df.values})

            # If there are less than 10 data points within a subset, then we skip this dataset (too small to be useful).
            if df.shape[0] < MIN_DATA_SIZE:
                continue

            df.to_csv(os.path.join(self.download_dir, f"{self.dataset_name}_{subset_name}.csv"))
