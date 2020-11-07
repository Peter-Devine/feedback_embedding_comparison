from scipy import spatial
from sklearn import metrics
import numpy as np

# Finds the average ROC over all points based on distance, where a 1 is defined as another point with a label that overlaps with the orginal point.
def calculate(embeddings, labels):
    distance_metric="euclidean"

    # Get the distances between each embedding
    mutual_distances = spatial.distance.squareform(spatial.distance.pdist(embeddings, metric=distance_metric))

    distances_sorted_args = [dist.argsort() for dist in mutual_distances]

    auc_score = 0

    for one_point_sorted_args in distances_sorted_args:
        # We first take out the point itself
        point_arg = one_point_sorted_args[0]
        one_point_sorted_args = one_point_sorted_args[1:]

        # We make a list with True if the point contains a label that the current reference point also has, and False if otherwise
        match_series = labels.iloc[one_point_sorted_args].apply(lambda x: any([lab in x for lab in labels.iloc[point_arg]])).values

        # We also make a series which has the same number of trues and falses as the match_series, but all the trues are at the start of the list
        # This would be the perfect embedding - which embeds all alike points closest first
        num_trues = match_series.sum()
        num_falses = len(match_series) - num_trues
        perfect_series = np.array([True] * num_trues + [False] * num_falses)

        print(perfect_series)
        print(match_series)

        fpr, tpr, thresholds = metrics.roc_curve(perfect_series, match_series, pos_label=2)
        auc_score += metrics.auc(fpr, tpr)

    average_auc = auc_score / len(distances_sorted_args)

    return average_auc
