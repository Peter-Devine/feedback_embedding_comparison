import pandas as pd
import os
import importlib

def get_metrics(list_of_metrics):

    if len(list_of_metrics) < 1:
        # If we have passed an empty list to the metric analyser, then we will calculate all metrics
        metrics_dir = os.path.join(".", "analysis", "metrics")
        metrics_file_list = os.listdir(metrics_dir)
        list_of_metrics = [file_name[:-3] for file_name in metrics_file_list if file_name[-3:] == ".py"]


    encodings_dir = os.path.join(".", "data", "encoding")

    results_dict = {}

    # Make an empty dict for each metric name to save dataset > encoding in it
    for metric_name in list_of_metrics:
        results_dict[metric_name] = {}

    # Since the encodings are located in ./data/encoding/DATASET_NAME/ENCODING_NAME.csv
    # we iterate over one folder then the other.
    for dataset_encodings_folder in os.listdir(encodings_dir):

        # Make an empty dict for each dataset in metric dict
        for metric_name, metric_dict in results_dict.items():
            metric_dict[dataset_encodings_folder] = {}

        datset_encodings_dir = os.path.join(encodings_dir, dataset_encodings_folder)
        dataset_encodings = [file for file in os.listdir(datset_encodings_dir) if file[-4:] == ".csv"]

        # Labels are the same across encodings, so we only read it once.
        # Get the labels column and parse the list of labels from string
        labels = get_labels(os.path.join(datset_encodings_dir, dataset_encodings[0]))

        # We prepare the metric objects so that we do not repeat label handling multiple times
        metric_objects_dict = prepare_metric_objects(list_of_metrics, labels)

        for encoding_file in dataset_encodings:

            # Load the encoding .csv
            encoding_dir = os.path.join(datset_encodings_dir, encoding_file)
            encoding_df = pd.read_csv(encoding_dir, index_col = 0)

            # Get encodings values
            encodings = encoding_df.drop("labels", axis=1).values

            print(f"Calculating metrics on {encoding_file} encoding of {dataset_encodings_folder}")
            metrics_dict = calculate_metrics_of_encoding(encodings, labels, metric_objects_dict)

            # So the results should be saved as {"metric_name": {"dataset_name": {"encoding_name": metric_val}}}
            for metric_name, metric_val in metrics_dict.items():
                results_dict[metric_name][dataset_encodings_folder][encoding_file] = metric_val

    RESULTS_DIR = os.path.join(".", "results")
    os.makedirs(RESULTS_DIR, exist_ok = True)
    for metric_name, metric_results_dict in results_dict.items():
        results_dir = os.path.join(RESULTS_DIR, f"{metric_name}.csv")
        pd.DataFrame(metric_results_dict).to_csv(results_dir)

def get_labels(encoding_path):
    encoding_df = pd.read_csv(encoding_path, index_col = 0)
    return encoding_df["labels"].apply(lambda x: x[1:-1].split(','))

def prepare_metric_objects(list_of_metrics, labels):
    metric_obj_dict = {}

    for metric_name in list_of_metrics:
        metric_module = importlib.import_module(f"analysis.metrics.{metric_name}")

        print(f"Loading {metric_name} metric")
        metric_obj = metric_module.Metric(labels)
        # If the metric is appliable to this label set
        if metric_obj.is_possible:
            metric_obj_dict[metric_name] = metric_obj

    return metric_obj_dict

def calculate_metrics_of_encoding(encodings, labels, metric_objects_dict):

    metrics_dict = {}

    for metric_name, metric_obj in metric_objects_dict.items():
        metric_val = metric_obj.calculate(encodings)
        metrics_dict[metric_name] = metric_val

    return metrics_dict
