import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(data):
    """
    Preprocess the urban infrastructure dataset
    """
    # Create a copy to avoid modifying original data
    processed_data = data.copy()
    
    # Handle missing values
    processed_data = processed_data.dropna()
    
    # Encode categorical variables
    if 'facility_type' in processed_data.columns:
        le = LabelEncoder()
        processed_data['facility_type_encoded'] = le.fit_transform(processed_data['facility_type'])
    
    # Ensure numeric types
    numeric_columns = ['year_established', 'latitude', 'longitude']
    for col in numeric_columns:
        if col in processed_data.columns:
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
    
    # Remove any rows with invalid coordinates or years
    processed_data = processed_data[
        (processed_data['latitude'].between(-90, 90)) &
        (processed_data['longitude'].between(-180, 180)) &
        (processed_data['year_established'] > 1800) &
        (processed_data['year_established'] <= 2024)
    ]
    
    return processed_data

def z_score_normalize(data):
    """
    Apply Z-score normalization as described in equation (1)
    z = (x - μ) / σ
    """
    if isinstance(data, pd.DataFrame):
        normalized_data = data.copy()
        for column in data.columns:
            mean_val = data[column].mean()
            std_val = data[column].std(ddof=0)
            
            # Avoid division by zero
            if std_val != 0:
                normalized_data[column] = (data[column] - mean_val) / std_val
            else:
                normalized_data[column] = 0
        
        return normalized_data.values
    else:
        # Handle numpy arrays
        normalized_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            column_data = data[:, i]
            mean_val = np.mean(column_data)
            std_val = np.std(column_data)
            
            if std_val != 0:
                normalized_data[:, i] = (column_data - mean_val) / std_val
            else:
                normalized_data[:, i] = 0
        
        return normalized_data

def create_temporal_splits(data, split_year=2010):
    """
    Split data into temporal subsets for analysis
    """
    pre_split = data[data['year_established'] < split_year]
    post_split = data[data['year_established'] >= split_year]
    
    return pre_split, post_split

def validate_dataset(data):
    """
    Validate that the dataset has required structure
    """
    required_columns = ['facility_type', 'year_established', 'latitude', 'longitude']
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for minimum data requirements
    if len(data) < 10:
        raise ValueError("Dataset must contain at least 10 records for meaningful clustering")
    
    # Check coordinate validity
    invalid_coords = data[
        ~data['latitude'].between(-90, 90) | 
        ~data['longitude'].between(-180, 180)
    ]
    
    if len(invalid_coords) > 0:
        raise ValueError(f"Found {len(invalid_coords)} records with invalid coordinates")
    
    return True
