import requests
import zipfile
import time
import json
import os
import io
import shutil

import pandas as pd

from data.download_util_base import DownloadUtilBase

class Downloader(DownloadUtilBase):

    def download(self):
        task_data_path = os.path.join(self.download_dir, "maalej_2016")
        os.makedirs(task_data_path, exist_ok = True)

        # from https://mast.informatik.uni-hamburg.de/wp-content/uploads/2015/06/review_classification_preprint.pdf
        # Bug Report, Feature Request, or Simply Praise? On Automatically Classifying App Reviews
        r = requests.get("https://mast.informatik.uni-hamburg.de/wp-content/uploads/2014/03/REJ_data.zip")
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(path = task_data_path)


        json_path = os.path.join(task_data_path, "REJ_data", "all.json")

        with open(json_path) as json_file:
            data = json.load(json_file)

        shutil.rmtree(task_data_path)

        df_data = [{"title": datum["title"],
                    "text": datum["comment"],
                    "rating": datum["rating"],
                    "label": datum["label"],
                    "reviewer": datum["reviewer"],
                    "app": datum["appId"],
                    "dataSource": datum["dataSource"],
                    "date": datum["date"]} for datum in data]

        df = pd.DataFrame(df_data)

        df_dict = {}

        for app_name in df.app.unique():
            # Get all observations that apply to this app
            app_df = df[df.app == app_name]

            # Drop all columns except text and label
            app_df = app_df.loc[:, ["text", "label"]]

            df_dict[app_name] = app_df

        return df_dict
