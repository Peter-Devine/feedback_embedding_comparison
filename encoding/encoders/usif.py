import os
import requests
import tarfile
import numpy as np

from encoding.usif_base import get_paranmt_usif

class Encoder:
    def __init__(self):

        self.usif_dir = os.path.join(".", "data", "usif")
        os.makedirs(self.usif_dir, exist_ok = True)

        self.download_paranmt()
        self.download_vocab()

        self.usif = get_paranmt_usif()

    def encode(self, text_list, dataset_name=None):
        return np.array(self.usif.embed(text_list))

    def download_paranmt(self):
        if not os.path.exists(os.path.join(self.usif_dir, "vectors", "czeng.txt")):
            temp_paranmt_dir = os.path.join(self.usif_dir, "temp.tar.gz")

            response = requests.get("https://raw.github.com/kawine/usif/master/paranmt.tar.gz", stream=True)
            if response.status_code == 200:
                with open(temp_paranmt_dir, 'wb') as f:
                    f.write(response.raw.read())
            else:
                raise Exception(f"Was not able to download ParaNMT in USIF because of error code {response.status_code}")

            tar = tarfile.open(temp_paranmt_dir, "r:gz")
            tar.extractall(path=self.usif_dir)
            tar.close()
            os.remove(temp_paranmt_dir)

    def download_vocab(self):
        vocab_dir = os.path.join(self.usif_dir, "enwiki_vocab_min200.txt")

        if not os.path.exists(vocab_dir):
            response = requests.get("https://raw.githubusercontent.com/kawine/usif/master/enwiki_vocab_min200.txt")
            with open(vocab_dir, 'w') as f:
                f.write(response.text)
