from scipy import spatial
from sklearn import metrics
import numpy as np

class Metric:

    # Because the labels are the same across encodings, we calculate the label similarity only once
    def __init__(self, labels):
        # Create a series of series which state whether the two elements at each index have any shared labels in their list of labels
        self.similarities = labels.apply(lambda outer_list: labels.apply(lambda inner_list: not set(outer_list).isdisjoint(set(inner_list)))).values

    # Finds the average ROC over all points based on distance, where a 1 is defined as another point with a label that overlaps with the orginal point.
    def calculate(self, embeddings):
        distance_metric="euclidean"

        # Get the distances between each embedding
        mutual_distances = spatial.distance.squareform(spatial.distance.pdist(embeddings, metric=distance_metric))

        print("euclidean")
        print(np.any(np.isnan(self.similarities)))
        print(np.any(np.isnan(mutual_distances)))

        sim_bins = [self.compare_closest(i, sim, dis) for i, (sim, dis) in enumerate(zip(self.similarities, -mutual_distances))]

        return sum(sim_bins) / len(sim_bins)

    def get_n_smallest_args(self, array, n):
        idx = np.argpartition(array, n)
        return idx[:n]

    # Calculate the roc auc score based on the similarity of labels (binary "yes, the two observations share at least one label" or "no, they share no labels")
    # The likelihood of them being classified as being similar is inversely proportional to the distance between them.
    def compare_closest(self, index, sim, dis):
        if sim.all():
            # In cases where a point has labels which cover ALL other points in the app (E.g. the feedback is marked as ['PROTECTION', 'OTHER', 'USAGE'] and
            # all other feedback has one of those labels), then we do not calculate the ROC AUC on it, but instead just return an AUC of 1, as literally any
            # feedback would be classed as relevant for this piece of feedback.
            return 1
        else:
            # We get the args of the two smallest values in the distance matrix because it will always be closest to itself.
            # We take the second values index
            lowest_two_args = self.get_n_smallest_args(dis, 2)
            closest_point_arg = lowest_two_args[lowest_two_args != index][0]
            return sim[closest_point_arg]
