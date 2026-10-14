import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

from algorithms.fuzzy_dbscan import FuzzyDBSCAN

class PSO:
    """
    Particle Swarm Optimization embedded in the fuzzy clustering loop.

    Every particle encodes the candidate parameter triple
    (eps, MinPtsMin, MinPtsMax). During each PSO iteration the candidate
    parameters are supplied directly to Fuzzy DBSCAN and the resulting
    clustering quality is evaluated with the Silhouette coefficient, which
    serves as the fitness function (Algorithm 2 of the manuscript).

    Search ranges follow Table 3 of the manuscript:
        eps          in [0.10, 1.00]
        MinPtsMin    in [2, 10]
        MinPtsMax    in [5, 20]

    Setting mode='dbscan' optimizes the classic two-parameter DBSCAN
    (eps, MinPts) instead; that variant is used as the PSO-DBSCAN baseline.
    """

    def __init__(self, n_particles=30, n_iterations=100, w=0.7, c1=2.0, c2=2.0,
                 eps_bounds=(0.10, 1.00), min_pts_min_bounds=(2, 10),
                 min_pts_max_bounds=(5, 20), mode='fuzzy',
                 convergence_patience=10, random_state=None,
                 min_clustered_fraction=0.3, max_cluster_fraction=0.1):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w            # inertia weight
        self.c1 = c1          # cognitive coefficient
        self.c2 = c2          # social coefficient
        self.eps_bounds = eps_bounds
        self.min_pts_min_bounds = min_pts_min_bounds
        self.min_pts_max_bounds = min_pts_max_bounds
        self.mode = mode
        self.convergence_patience = convergence_patience
        # Degenerate-solution guard: reject configurations that mark too many
        # points as noise or shatter the data into micro-clusters, which would
        # otherwise game the Silhouette fitness.
        self.min_clustered_fraction = min_clustered_fraction
        self.max_cluster_fraction = max_cluster_fraction
        self.rng = np.random.default_rng(random_state)

        if mode == 'fuzzy':
            self.bounds = [eps_bounds, min_pts_min_bounds, min_pts_max_bounds]
        elif mode == 'dbscan':
            self.bounds = [eps_bounds, (2, 20)]
        else:
            raise ValueError("mode must be 'fuzzy' or 'dbscan'")

        self.bounds = np.array(self.bounds, dtype=float)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize_particles(self):
        """Initialize particle positions and velocities within the bounds."""
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        positions = self.rng.uniform(low=low, high=high,
                                     size=(self.n_particles, len(self.bounds)))
        if self.mode == 'fuzzy':
            self._enforce_min_pts_order(positions)

        velocities = self.rng.uniform(
                low=-0.1 * (high - low), high=0.1 * (high - low),
                size=(self.n_particles, len(self.bounds)))
        return positions, velocities

    @staticmethod
    def _enforce_min_pts_order(positions):
        """Guarantee MinPtsMin < MinPtsMax for every particle."""
        min_pts_min = positions[:, 1]
        min_pts_max = positions[:, 2]
        swap = min_pts_min >= min_pts_max
        if np.any(swap):
            positions[swap, 1], positions[swap, 2] = (
                positions[swap, 2].copy(), positions[swap, 1].copy())

    # ------------------------------------------------------------------
    # Fitness: fuzzy DBSCAN executed with the candidate parameters
    # ------------------------------------------------------------------
    def evaluate_fitness(self, data, params):
        """
        Execute clustering with the candidate parameters and score it with
        the Silhouette coefficient (fitness function of Algorithm 2).
        Returns the Silhouette score, or -1.0 for degenerate solutions.
        """
        try:
            labels = self._cluster(data, params)
            if self._is_degenerate(labels):
                return -1.0
            return self._silhouette(data, labels)
        except Exception:
            return -1.0

    def _is_degenerate(self, labels):
        """Reject solutions with too much noise or only micro-clusters."""
        mask = labels != -1
        n_clustered = int(np.sum(mask))
        if n_clustered < 2:
            return True
        if n_clustered < self.min_clustered_fraction * len(labels):
            return True
        n_clusters = len(np.unique(labels[mask]))
        if n_clusters < 2:
            return True
        return n_clusters > max(2, self.max_cluster_fraction * len(labels))

    @staticmethod
    def _silhouette(data, labels):
        mask = labels != -1
        if np.sum(mask) < 2:
            return -1.0
        unique_labels = np.unique(labels[mask])
        if len(unique_labels) < 2:
            return -1.0
        try:
            return float(silhouette_score(data[mask], labels[mask]))
        except Exception:
            return -1.0

    def _cluster(self, data, params):
        if self.mode == 'fuzzy':
            eps, min_pts_min, min_pts_max = params
            min_pts_min = int(round(min_pts_min))
            min_pts_max = int(round(max(min_pts_max, min_pts_min + 1)))
            model = FuzzyDBSCAN(eps=eps, min_pts_min=min_pts_min,
                                min_pts_max=min_pts_max)
            labels, _ = model.fit_predict(data)
            return labels

        eps, min_pts = params
        min_pts = max(2, int(round(min_pts)))
        return DBSCAN(eps=eps, min_samples=min_pts).fit_predict(data)

    # ------------------------------------------------------------------
    # Velocity / position updates (standard PSO formulation)
    # ------------------------------------------------------------------
    def update_velocity(self, velocity, position, personal_best, global_best, r1, r2):
        cognitive = self.c1 * r1 * (personal_best - position)
        social = self.c2 * r2 * (global_best - position)
        new_velocity = self.w * velocity + cognitive + social

        # Velocity clamping: 10% of each dimension's search range
        max_velocity = 0.1 * (self.bounds[:, 1] - self.bounds[:, 0])
        return np.clip(new_velocity, -max_velocity, max_velocity)

    def update_position(self, position, velocity):
        new_position = position + velocity
        np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1],
                out=new_position)
        if self.mode == 'fuzzy':
            self._enforce_min_pts_order(new_position.reshape(1, -1))
        return new_position

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def optimize(self, data):
        """
        PSO main loop (Algorithm 2). Returns the global best parameter set
        together with its fitness value and the per-iteration history.
        """
        data = np.asarray(data, dtype=float)
        positions, velocities = self.initialize_particles()

        personal_best_positions = positions.copy()
        personal_best_fitness = np.full(self.n_particles, -np.inf)
        global_best_position = None
        global_best_fitness = -np.inf
        history = []
        stale_iterations = 0

        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                fitness = self.evaluate_fitness(data, positions[i])

                if fitness > personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_positions[i] = positions[i].copy()

                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = positions[i].copy()

            history.append({
                'iteration': iteration,
                'fitness': global_best_fitness,
                'params': self._format_params(global_best_position)
            })

            # Stopping criterion: convergence (no improvement for
            # `convergence_patience` iterations) or maximum iterations.
            if iteration > 0 and history[-1]['fitness'] <= history[-2]['fitness']:
                stale_iterations += 1
                if stale_iterations >= self.convergence_patience:
                    break
            else:
                stale_iterations = 0

            for i in range(self.n_particles):
                r1, r2 = self.rng.random(2)
                velocities[i] = self.update_velocity(
                    velocities[i], positions[i],
                    personal_best_positions[i], global_best_position, r1, r2)
                positions[i] = self.update_position(positions[i], velocities[i])

        return self._format_params(global_best_position, global_best_fitness, history)

    def _format_params(self, position, fitness=-1.0, history=None):
        if position is None:
            # Fallback: reasonable defaults if optimization produced nothing
            if self.mode == 'fuzzy':
                return {'eps': 0.5, 'min_pts_min': 3, 'min_pts_max': 10,
                        'fitness': -1.0, 'history': history or []}
            return {'eps': 0.5, 'min_pts': 5, 'fitness': -1.0,
                    'history': history or []}

        if self.mode == 'fuzzy':
            return {
                'eps': float(position[0]),
                'min_pts_min': int(round(position[1])),
                'min_pts_max': int(round(position[2])),
                'fitness': float(fitness),
                'history': history or []
            }
        return {
            'eps': float(position[0]),
            'min_pts': int(round(position[1])),
            'fitness': float(fitness),
            'history': history or []
        }
