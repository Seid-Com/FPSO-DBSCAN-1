import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def calculate_performance_metrics(data, labels):
    """
    Calculate clustering performance metrics
    Primary metric is Silhouette Score as used in the research paper
    """
    # Filter out noise points (-1 labels)
    mask = labels != -1
    
    if np.sum(mask) < 2:
        return -1.0  # Return poor score if insufficient valid points
    
    filtered_data = data[mask]
    filtered_labels = labels[mask]
    
    # Check if we have at least 2 clusters
    unique_labels = np.unique(filtered_labels)
    if len(unique_labels) < 2:
        return -1.0
    
    try:
        silhouette = silhouette_score(filtered_data, filtered_labels)
        return silhouette
    except Exception:
        return -1.0

def calculate_additional_metrics(data, labels):
    """
    Calculate additional clustering metrics for comprehensive evaluation
    """
    metrics = {}
    
    # Filter out noise points
    mask = labels != -1
    
    if np.sum(mask) < 2:
        return {
            'silhouette_score': -1.0,
            'calinski_harabasz_score': -1.0,
            'davies_bouldin_score': float('inf'),
            'n_clusters': 0,
            'n_noise_points': np.sum(~mask),
            'clustering_efficiency': 0.0
        }
    
    filtered_data = data[mask]
    filtered_labels = labels[mask]
    
    unique_labels = np.unique(filtered_labels)
    n_clusters = len(unique_labels)
    
    try:
        # Silhouette Score
        silhouette = silhouette_score(filtered_data, filtered_labels)
        metrics['silhouette_score'] = silhouette
    except Exception:
        metrics['silhouette_score'] = -1.0
    
    try:
        # Calinski-Harabasz Score (higher is better)
        ch_score = calinski_harabasz_score(filtered_data, filtered_labels)
        metrics['calinski_harabasz_score'] = ch_score
    except Exception:
        metrics['calinski_harabasz_score'] = -1.0
    
    try:
        # Davies-Bouldin Score (lower is better)
        db_score = davies_bouldin_score(filtered_data, filtered_labels)
        metrics['davies_bouldin_score'] = db_score
    except Exception:
        metrics['davies_bouldin_score'] = float('inf')
    
    # Basic clustering statistics
    metrics['n_clusters'] = n_clusters
    metrics['n_noise_points'] = np.sum(~mask)
    metrics['clustering_efficiency'] = np.sum(mask) / len(data)
    
    return metrics

def statistical_validation(fuzzy_scores, standard_scores):
    """
    Perform statistical validation as mentioned in the research paper
    Uses paired t-test to validate performance differences
    """
    if len(fuzzy_scores) != len(standard_scores):
        raise ValueError("Score arrays must have the same length")
    
    if len(fuzzy_scores) < 2:
        return {
            'test_type': 'insufficient_data',
            't_statistic': None,
            'p_value': None,
            'significant': False,
            'interpretation': 'Insufficient data for statistical test'
        }
    
    # Perform paired t-test
    t_statistic, p_value = stats.ttest_rel(fuzzy_scores, standard_scores)
    
    # Determine significance (α = 0.05)
    significant = p_value < 0.05
    
    # Effect size (Cohen's d for paired samples)
    differences = np.array(fuzzy_scores) - np.array(standard_scores)
    effect_size = np.mean(differences) / np.std(differences) if np.std(differences) != 0 else 0
    
    # Interpretation
    if significant:
        if np.mean(fuzzy_scores) > np.mean(standard_scores):
            interpretation = "Fuzzy-PSO DBSCAN significantly outperforms standard DBSCAN"
        else:
            interpretation = "Standard DBSCAN significantly outperforms Fuzzy-PSO DBSCAN"
    else:
        interpretation = "No significant difference between methods"
    
    return {
        'test_type': 'paired_t_test',
        't_statistic': t_statistic,
        'p_value': p_value,
        'significant': significant,
        'effect_size': effect_size,
        'mean_fuzzy': np.mean(fuzzy_scores),
        'mean_standard': np.mean(standard_scores),
        'mean_improvement': np.mean(differences),
        'interpretation': interpretation
    }

def calculate_cluster_statistics(data, labels, memberships=None):
    """
    Calculate detailed statistics for each cluster
    """
    unique_labels = np.unique(labels)
    cluster_stats = {}
    
    for label in unique_labels:
        if label == -1:  # Noise points
            cluster_stats['noise'] = {
                'size': np.sum(labels == -1),
                'percentage': (np.sum(labels == -1) / len(labels)) * 100
            }
            continue
        
        cluster_mask = labels == label
        cluster_data = data[cluster_mask]
        
        stats_dict = {
            'size': len(cluster_data),
            'percentage': (len(cluster_data) / len(data)) * 100,
            'centroid': np.mean(cluster_data, axis=0),
            'std_dev': np.std(cluster_data, axis=0),
            'diameter': calculate_cluster_diameter(cluster_data)
        }
        
        # Add fuzzy membership statistics if available
        if memberships is not None:
            cluster_memberships = [memberships[i][label] for i in range(len(memberships)) 
                                 if label < len(memberships[i]) and labels[i] == label]
            if cluster_memberships:
                stats_dict['avg_membership'] = np.mean(cluster_memberships)
                stats_dict['min_membership'] = np.min(cluster_memberships)
                stats_dict['max_membership'] = np.max(cluster_memberships)
        
        cluster_stats[f'cluster_{label}'] = stats_dict
    
    return cluster_stats

def calculate_cluster_diameter(cluster_data):
    """
    Calculate the diameter of a cluster (maximum distance between any two points)
    """
    if len(cluster_data) < 2:
        return 0.0
    
    max_distance = 0.0
    for i in range(len(cluster_data)):
        for j in range(i + 1, len(cluster_data)):
            distance = np.linalg.norm(cluster_data[i] - cluster_data[j])
            max_distance = max(max_distance, float(distance))
    
    return max_distance

def evaluate_parameter_sensitivity(data, eps_range, min_pts_range):
    """
    Evaluate parameter sensitivity for DBSCAN
    Useful for understanding the robustness of PSO optimization
    """
    sensitivity_results = []
    
    for eps in eps_range:
        for min_pts in min_pts_range:
            try:
                from sklearn.cluster import DBSCAN
                dbscan = DBSCAN(eps=eps, min_samples=min_pts)
                labels = dbscan.fit_predict(data)
                
                score = calculate_performance_metrics(data, labels)
                
                sensitivity_results.append({
                    'eps': eps,
                    'min_pts': min_pts,
                    'silhouette_score': score,
                    'n_clusters': len(np.unique(labels)) - (1 if -1 in labels else 0),
                    'n_noise': np.sum(labels == -1)
                })
            except Exception:
                continue
    
    return sensitivity_results
