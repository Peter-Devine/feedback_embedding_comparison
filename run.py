from data.downloader import get_data

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--download_list', default="", required=False, type=str, help='Comma separated list of datasets to download.')
parser.add_argument('--download_all', type=bool, nargs='?', const=True, default=False, help="Download all data available?")
# parser.add_argument('--LR', default=5e-5, type=int, help='Learning rate for the model')

args = parser.parse_args()

print(f"Inputted args are:\n{args}")
if len(args.download_list) > 0 or args.download_all:
    dataset_list = [] if args.download_all else args.download_list.split(",")
    get_data(dataset_list)
