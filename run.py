from data.downloader import get_data
from encoding.encode import get_encodings
from analysis.analyse import get_metrics

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--download_list', default="", required=False, type=str, help='Comma separated list of datasets to download.')
parser.add_argument('--download_all', type=bool, nargs='?', const=True, default=False, help="Download all data available?")
parser.add_argument('--encoding_list', default="", required=False, type=str, help="Comma separated list of encoders to encode with.")
parser.add_argument('--encode_all', type=bool, nargs='?', const=True, default=False, help="Encode using all encoders available?")
parser.add_argument('--metric_list', default="", required=False, type=str, help="Comma separated list of metrics to analyse encodings with.")
parser.add_argument('--calculate_all_metrics', type=bool, nargs='?', const=True, default=False, help="Analyse using all metrics available?")
parser.add_argument('--eval_encoding_name', default="", required=False, type=str, help="Which encodings should we calculate metrics for? If blank, defaults to all currently generated encodings.")
# parser.add_argument('--LR', default=5e-5, type=int, help='Learning rate for the model')

args = parser.parse_args()

print(f"Inputted args are:\n{args}")
if len(args.download_list) > 0 or args.download_all:
    dataset_list = [] if args.download_all else args.download_list.split(",")
    get_data(dataset_list)

if len(args.encoding_list) > 0 or args.encode_all:
    encoders_list = [] if args.encode_all else args.encoding_list.split(",")
    get_encodings(encoders_list)

if len(args.metric_list) > 0 or args.calculate_all_metrics:
    metric_list = [] if args.calculate_all_metrics else args.metric_list.split(",")
    eval_encoding_list = [] if len(args.eval_encoding_name) < 1 else args.eval_encoding_name.split(",")
    get_metrics(metric_list, eval_encoding_list)
