import pytest
import pandas as pd
import numpy as np
from algorithms.pso import PSO
from algorithms.fuzzy_dbscan import FuzzyDBSCAN
from utils.data_processing import preprocess_data, z_score_normalize
from utils.metrics import calculate_performance_metrics, calculate_additional_metrics

@pytest.fixture
def sample_data():
    """Create sample urban infrastructure data for testing"""
    return pd.DataFrame({
        'facility_type': ['education', 'health', 'entertainment', 'education', 'health'],
        'year_established': [2005, 2012, 2008, 2015, 2003],
        'latitude': [9.0307, 9.0397, 9.0257, 9.0407, 9.0287],
        'longitude': [38.7407, 38.7507, 38.7357, 38.7557, 38.7387]
    })

def test_data_preprocessing(sample_data):
    """Test data preprocessing functionality"""
    processed = preprocess_data(sample_data)

    assert 'facility_type_encoded' in processed.columns
    assert len(processed) == 5
    assert processed['facility_type_encoded'].dtype in ['int32', 'int64']

def test_z_score_normalization():
    """Test Z-score normalization (equation 1)"""
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    normalized = z_score_normalize(pd.DataFrame(data))

    # Check that normalized data has mean ~0 and std ~1
    assert abs(np.mean(normalized)) < 1e-10
    assert abs(np.std(normalized) - 1.0) < 1e-10

def test_fuzzy_core_membership_equation():
    """Fuzzy-core membership function must follow equation (2)"""
    model = FuzzyDBSCAN(eps=0.5, min_pts_min=3, min_pts_max=7)

    assert model.fuzzy_core_membership(2) == 0.0    # n <= MinPtsMin
    assert model.fuzzy_core_membership(3) == 0.0    # n == MinPtsMin
    assert model.fuzzy_core_membership(7) == 1.0    # n >= MinPtsMax
    assert model.fuzzy_core_membership(10) == 1.0   # n > MinPtsMax
    # Intermediate value: (5 - 3) / (7 - 3) = 0.5
    assert abs(model.fuzzy_core_membership(5) - 0.5) < 1e-12

def test_fuzzy_dbscan_parameter_validation():
    """MinPtsMin must be strictly smaller than MinPtsMax"""
    with pytest.raises(ValueError):
        FuzzyDBSCAN(eps=0.5, min_pts_min=7, min_pts_max=7)

def test_pso_fuzzy_optimization():
    """PSO must optimize the parameter triple (eps, MinPtsMin, MinPtsMax)"""
    np.random.seed(42)
    cluster1 = np.random.normal([0, 0], 0.1, (30, 2))
    cluster2 = np.random.normal([2, 2], 0.1, (30, 2))
    data = np.vstack([cluster1, cluster2])

    pso = PSO(n_particles=5, n_iterations=5, mode='fuzzy', random_state=42)
    result = pso.optimize(data)

    assert 'eps' in result
    assert 'min_pts_min' in result
    assert 'min_pts_max' in result
    assert 'fitness' in result
    assert 0.10 <= result['eps'] <= 1.00          # Table 3 search range
    assert 2 <= result['min_pts_min'] <= 10
    assert 5 <= result['min_pts_max'] <= 20
    assert result['min_pts_min'] < result['min_pts_max']
    assert len(result['history']) >= 1

def test_pso_dbscan_mode():
    """PSO-DBSCAN baseline mode optimizes classic (eps, MinPts)"""
    np.random.seed(42)
    data = np.random.rand(40, 3)

    pso = PSO(n_particles=5, n_iterations=5, mode='dbscan', random_state=42)
    result = pso.optimize(data)

    assert 'eps' in result
    assert 'min_pts' in result
    assert 0.10 <= result['eps'] <= 1.00

def test_fuzzy_dbscan():
    """Test Fuzzy DBSCAN clustering (Algorithm 1)"""
    np.random.seed(42)
    cluster1 = np.random.normal([0, 0], 0.1, (10, 2))
    cluster2 = np.random.normal([2, 2], 0.1, (10, 2))
    data = np.vstack([cluster1, cluster2])

    model = FuzzyDBSCAN(eps=0.3, min_pts_min=3, min_pts_max=6)
    labels, memberships = model.fit_predict(data)

    assert len(labels) == 20
    assert len(memberships) == 20
    assert len(set(labels) - {-1}) >= 2  # Should find at least 2 clusters
    # Membership degrees lie in [0, 1] and noise points have membership 0
    assert np.all(memberships >= 0) and np.all(memberships <= 1)
    assert np.all(memberships[labels == -1] == 0)
    # Clustered points have positive membership
    assert np.all(memberships[labels != -1] > 0)

def test_fuzzy_dbscan_membership_matrix():
    """fit_predict_matrix returns one membership column per cluster"""
    np.random.seed(0)
    data = np.vstack([np.random.normal([0, 0], 0.1, (15, 2)),
                      np.random.normal([3, 3], 0.1, (15, 2))])

    model = FuzzyDBSCAN(eps=0.5, min_pts_min=3, min_pts_max=6)
    labels, matrix = model.fit_predict_matrix(data)

    n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
    assert matrix.shape == (30, n_clusters)
    for i, label in enumerate(labels):
        if label >= 0:
            assert matrix[i, label] > 0

def test_performance_metrics():
    """Test performance metrics calculation"""
    data = np.array([[0, 0], [0, 1], [1, 0], [5, 5], [5, 6], [6, 5]])
    labels = np.array([0, 0, 0, 1, 1, 1])

    score = calculate_performance_metrics(data, labels)

    assert -1.0 <= score <= 1.0
    assert isinstance(score, float)

    extra = calculate_additional_metrics(data, labels)
    assert extra['n_clusters'] == 2
    assert extra['silhouette_score'] > 0
    assert 'calinski_harabasz_score' in extra
    assert 'davies_bouldin_score' in extra

def test_invalid_data_handling():
    """Test handling of invalid data"""
    invalid_data = pd.DataFrame({
        'facility_type': ['education'],
        'year_established': [2005],
        'latitude': [91],  # Invalid latitude
        'longitude': [38.7407]
    })

    processed = preprocess_data(invalid_data)
    assert len(processed) == 0  # Should filter out invalid coordinates

if __name__ == "__main__":
    pytest.main([__file__])
