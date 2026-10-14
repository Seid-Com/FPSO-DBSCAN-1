import numpy as np
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

class FuzzyDBSCAN:
    """
    Fuzzy DBSCAN implementation following Algorithm 1 of the manuscript
    "Fuzzy-PSO Optimized DBSCAN for Non-Spatial Clustering".

    Replaces the classic MinPts parameter with two parameters,
    MinPtsMin and MinPtsMax, and assigns a fuzzy membership degree to
    every clustered point through the fuzzy-core membership function
    (equation 2):

        mu_minp(n) = 1                                         if n >= MinPtsMax
                   = (n - MinPtsMin) / (MinPtsMax - MinPtsMin)  if MinPtsMin < n < MinPtsMax
                   = 0                                         if n <= MinPtsMin
    """

    def __init__(self, eps=0.5, min_pts_min=3, min_pts_max=10):
        if min_pts_min >= min_pts_max:
            raise ValueError("MinPtsMin must be strictly smaller than MinPtsMax")
        self.eps = eps
        self.min_pts_min = int(min_pts_min)
        self.min_pts_max = int(min_pts_max)

    def fuzzy_core_membership(self, n):
        """Fuzzy-core membership function mu_minp(n), equation (2)."""
        if n <= self.min_pts_min:
            return 0.0
        if n >= self.min_pts_max:
            return 1.0
        return (n - self.min_pts_min) / (self.min_pts_max - self.min_pts_min)

    def _neighborhoods(self, data):
        """Precompute the eps-neighborhood of every point (regionQuery)."""
        distances = cdist(data, data)
        return [np.flatnonzero(row <= self.eps) for row in distances]

    def _expand_cluster(self, neighborhoods, seed_idx, cluster_id, labels, visited, memberships):
        """
        Expand a cluster over its density-reachable points (Algorithm 1).
        """
        labels[seed_idx] = cluster_id
        memberships[seed_idx] = self.fuzzy_core_membership(len(neighborhoods[seed_idx]))

        processed = {int(seed_idx)}
        queue = [int(n) for n in neighborhoods[seed_idx]]
        i = 0

        while i < len(queue):
            neighbor_idx = queue[i]
            i += 1

            if neighbor_idx in processed:
                continue
            processed.add(neighbor_idx)

            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                neighbor_neighbors = neighborhoods[neighbor_idx]
                # Fuzzy core point: mu_minp(|N|) > 0  <=>  |N| > MinPtsMin.
                # Such a point extends the cluster with its own neighbors.
                if len(neighbor_neighbors) > self.min_pts_min:
                    for nn in neighbor_neighbors:
                        nn = int(nn)
                        if nn not in processed:
                            queue.append(nn)

            # Border point: density reachable but not a fuzzy core point.
            # It still joins the cluster with its own membership degree.
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
                memberships[neighbor_idx] = self.fuzzy_core_membership(
                    len(neighborhoods[neighbor_idx]))

    def fit_predict(self, data):
        """
        Fuzzy DBSCAN clustering (Algorithm 1).

        Returns
        -------
        labels : np.ndarray
            Crisp cluster assignment per point (-1 for noise), obtained from
            the fuzzy assignment (argmax membership).
        memberships : np.ndarray
            Membership degree of each point in its assigned cluster
            (0 for noise points).
        """
        data = np.asarray(data, dtype=float)
        n_points = len(data)
        labels = np.full(n_points, -1, dtype=int)
        memberships = np.zeros(n_points, dtype=float)
        visited = np.full(n_points, False)
        cluster_id = 0

        neighborhoods = self._neighborhoods(data)

        for i in range(n_points):
            if visited[i]:
                continue
            visited[i] = True

            neighbors = neighborhoods[i]

            # Algorithm 1: noise when the neighborhood holds no more than
            # MinPtsMin points (mu_minp(n) == 0).
            if len(neighbors) <= self.min_pts_min:
                labels[i] = -1
                memberships[i] = 0.0
                continue

            cluster_id += 1
            self._expand_cluster(neighborhoods, i, cluster_id, labels,
                                 visited, memberships)

        # Shift labels so clusters are numbered from 0 (noise stays -1)
        labels = np.where(labels > 0, labels - 1, labels)
        return labels, memberships

    def fit_predict_matrix(self, data):
        """
        Variant returning a per-cluster membership matrix (one column per
        cluster). Used by the visualization layer.
        """
        labels, memberships = self.fit_predict(data)
        n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
        matrix = np.zeros((len(labels), n_clusters))
        for i, label in enumerate(labels):
            if label >= 0:
                matrix[i, label] = memberships[i]
        return labels, matrix
