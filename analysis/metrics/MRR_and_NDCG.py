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

        distance_metrics = ["cosine", "euclidean", "jensenshannon", "manhattan"]
        results_dict = {}

        for distance_metric in distance_metrics:
            # Get the distances between each embedding
            mutual_distances = spatial.distance.squareform(spatial.distance.pdist(embeddings, metric=distance_metric))

            # Make the nan and inf values (which only appear in cosine and JS distances) 2, higher than the maximum for these metrics
            mutual_distances[np.isnan(mutual_distances) | np.isinf(mutual_distances)] = 2

            list_of_rr = [self.__calculate_metric(self.mrr, sim, dis, i) for i, (sim, dis) in enumerate(zip(self.similarities, mutual_distances))]
            list_of_ndcg = [self.__calculate_metric(self.ndcg, sim, dis, i) for i, (sim, dis) in enumerate(zip(self.similarities, mutual_distances))]
            list_of_rr = [x for x in list_of_rr if x is not None]
            list_of_ndcg = [x for x in list_of_ndcg if x is not None]

            mrr = statistics.mean(list_of_rr)
            ndcg = statistics.mean(list_of_ndcg)

            results_dict.update({f"{distance_metric}_mrr": mrr, f"{distance_metric}_ndcg": ndcg})

        return results_dict

    def __calculate_metric(self, metric_fn, pos_neg_vector, distances, idx):
        # Remove the point itself from the comparison
        dis = np.delete(distances, idx)
        p_n_vec = np.delete(pos_neg_vector, idx)

        # If all points are the same, or none are, then rankings of other feedback is meaningless.
        # Thus we return None, and exclude these points from the calculation.
        if p_n_vec.all() or not p_n_vec.any():
            return None
        else:
            sorted_args = np.argsort(dis)
            return metric_fn.compute(p_n_vec, sorted_args)
