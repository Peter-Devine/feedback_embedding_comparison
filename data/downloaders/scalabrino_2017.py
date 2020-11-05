import pandas as pd
from data.download_util_base import DownloadUtilBase

class Downloader(DownloadUtilBase):
    def download(self):

        df = pd.read_csv("https://dibt.unimol.it/reports/clap/downloads/rq3-manually-classified-implemented-reviews.csv")

        df = df.rename(columns = {"body": "text", "category": "label"})
        df["app"] = df["App-name"]
        df["sublabel"] = df["rating"]

        df_dict = {}

        for app_name in df.app.unique():
            # Get all observations that apply to this app
            app_df = df[df.app == app_name]

            # Drop all columns except text and label
            app_df = app_df.loc[:, ["text", "label"]]

            df_dict[app_name] = app_df

        return df_dict
