from scipy import spatial
from sklearn import metrics
import numpy as np

class Metric:

    # Because the labels are the same across encodings, we calculate the label similarity only once
    def __init__(self, labels):
        # Create a series of series which state whether the two elements at each index have any shared labels in their list of labels
        self.similarities = labels.apply(lambda outer_list: labels.apply(lambda inner_list: not set(outer_list).isdisjoint(set(inner_list)))).values

        # If all values are True, then there are no different labels in the dataset (all points share at least one label with every other point)
        # In this instance, we set self.is_all_true to True, and analyse.py will  as the ROC curve would then be meaningless
        self.is_possible = not self.similarities.all().all()

    # Finds the average ROC over all points based on distance, where a 1 is defined as another point with a label that overlaps with the orginal point.
    def calculate(self, embeddings):
        distance_metric="euclidean"

        # Get the distances between each embedding
        mutual_distances = spatial.distance.squareform(spatial.distance.pdist(embeddings, metric=distance_metric))

        print("euclidean")
        print(np.any(np.isnan(self.similarities)))
        print(np.any(np.isnan(mutual_distances)))
        # Calculate the roc auc score based on the similarity of labels (binary "yes, the two observations share at least one label" or "no, they share no labels")
        # The likelihood of them being classified as being similar is inversely proportional to the distance between them.
        def calculate_rocauc(sim, dis):
            if sim.all():
                # In cases where a point has labels which cover ALL other points in the app (E.g. the feedback is marked as ['PROTECTION', 'OTHER', 'USAGE'] and
                # all other feedback has one of those labels), then we do not calculate the ROC AUC on it, but instead just return an AUC of 1, as literally any
                # feedback would be classed as relevant for this piece of feedback.
                return 1
            else:
                return metrics.roc_auc_score(sim, dis)

        rocs = [calculate_rocauc(sim, dis, i) for i, (sim, dis) in enumerate(zip(self.similarities, -mutual_distances))]

        return sum(rocs) / len(rocs)
