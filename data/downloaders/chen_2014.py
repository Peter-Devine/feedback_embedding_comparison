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

        task_data_path = os.path.join(self.download_dir, "chen_2014")
        os.makedirs(task_data_path, exist_ok = True)
        # from https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=3323&context=sis_research
        # AR-Miner: Mining Informative Reviews for Developers from Mobile App Marketplace
        r = requests.get("https://sites.google.com/site/appsuserreviews/home/datasets.zip?attredirects=0&d=1")
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(path=task_data_path)

        def get_splits_per_app(app_name):
            def df_getter(data_path, label):
                with open(data_path, "r", encoding="iso-8859-1") as f:
                    data = f.read()

                reviews = [" ".join(x.split()[2:]) for x in data.split("\n") if len(x) > 0]
                ratings = [x.split()[1] for x in data.split("\n") if len(x) > 0]

                return pd.DataFrame({"text": reviews,
                                     "label": label})

            train_info = df_getter(os.path.join(task_data_path, "datasets", app_name, "trainL", "info.txt"), "informative")
            train_noninfo = df_getter(os.path.join(task_data_path, "datasets", app_name, "trainL", "non-info.txt"), "non-informative")

            test_info = df_getter(os.path.join(task_data_path, "datasets", app_name, "test", "info.txt"), "informative")
            test_noninfo = df_getter(os.path.join(task_data_path, "datasets", app_name, "test", "non-info.txt"), "non-informative")

            train = train_info.append(train_noninfo).reset_index(drop=True)
            test = test_info.append(test_noninfo).reset_index(drop=True)

            ## Handles a bug in 3 of the datasets where a piece of feedback is labelled as both informative and non-informative
            bad_texts = ["Frustrating since I had to delete all of the learned suggestions, but it's back so I'm happy.",
                        "cute game but have to play it every day or fish will die of starvation",
                        "give more fish bucks and lower prices, but it was awsome",
                        "now ive lot all my prizes that's again tap fish, you're getting deleted finally",
                        "why is tapfish disabled",
                        "but when are you guys gonna have better graphics its 2013",
                        "excellent game but new upgrade 3d version or hd version"]
            train = train.loc[~train.text.isin(bad_texts),:]
            test = test.loc[~test.text.isin(bad_texts),:]

            return train, test

        def append_df(df1, df2):
            return df1.append(df2).reset_index(drop=True)

        df_dict = {}
        for app_name in ["swiftkey", "facebook", "tapfish", "templerun2"]:
            app_train, app_test = get_splits_per_app(app_name)
            app_df = append_df(app_train, app_test)
            df_dict[app_name] = app_df

        shutil.rmtree(task_data_path)

        return df_dict
