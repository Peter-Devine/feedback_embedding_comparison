from scipy import spatial
from sklearn import metrics
import numpy as np

import statistics

from cornac.metrics import NDCG, MRR

class Metric:

    # Because the labels are the same across encodings, we calculate the label similarity only once
    def __init__(self, labels):
        # Create a series of series which state whether the two elements at each index have any shared labels in their list of labels
        self.similarities = labels.apply(lambda outer_list: labels.apply(lambda inner_list: not set(outer_list).isdisjoint(set(inner_list)))).values
        self.mrr = MRR()
        self.ndcg = NDCG()

    # Finds the average ROC over all points based on distance, where a 1 is defined as another point with a label that overlaps with the orginal point.
    def calculate(self, embeddings):

        distance_metrics = ["cosine", "euclidean", "jensenshannon"]
        results_dict = {}

        for distance_metric in distance_metrics:
            # Get the distances between each embedding
            mutual_distances = spatial.distance.squareform(spatial.distance.pdist(embeddings, metric=distance_metric))

            # Make the nan and inf values (which only appear in cosine and JS distances) 2, higher than the maximum for these metrics
            mutual_distances[np.isnan(mutual_distances) | np.isinf(mutual_distances)] = 2

            list_of_rr = [self.__calculate_metric(self.mrr, sim, dis) for sim, dis in zip(self.similarities, mutual_distances)]
            list_of_ndcg = [self.__calculate_metric(self.ndcg, sim, dis) for sim, dis in zip(self.similarities, mutual_distances)]

            mrr = statistics.mean(list_of_rr)
            ndcg = statistics.mean(list_of_ndcg)

            results_dict.update({f"{distance_metric}_mrr": mrr, f"{distance_metric}_ndcg": ndcg})

        return results_dict

    def __calculate_metric(self, metric_fn, pos_neg_vector, distances):
        # If this point is similar to all points (I.e. shares a label with every other piece of feedback) then we just return 1, as anything you return will be appropriate
        if pos_neg_vector.all():
            return 1
        else:
            sorted_args = np.argsort(distances)
            return metric_fn.compute(pos_neg_vector, sorted_args)
