#Fuzzy–PSO Optimized DBSCAN for Non-Spatial Clustering

A Streamlit-based web application implementing the hybrid fuzzy–particle swarm optimization DBSCAN framework from the manuscript *"Fuzzy–PSO Optimized DBSCAN for Non-Spatial Clustering"* by Seid Mehammed Abdu (Woldia University, Ethiopia).

## Features

- **Interactive Web Interface**: Built with Streamlit for easy data upload and parameter configuration
- **PSO embedded in the fuzzy loop**: PSO continuously optimizes ε, MinPtsMin and MinPtsMax while fuzzy DBSCAN runs, using the Silhouette coefficient as fitness (Algorithm 2)
- **Fuzzy membership (Eq. 2)**: points near cluster boundaries receive intermediate membership degrees instead of crisp assignment
- **Non-spatial clustering**: clusters the normalized attributes (year of establishment, numeric service type); GPS coordinates are used only for GIS display
- **Four-method comparison (Table 4)**: DBSCAN, Fuzzy DBSCAN, PSO-DBSCAN and the proposed Fuzzy–PSO, scored with Silhouette, Davies-Bouldin, Calinski-Harabasz and execution time
- **Statistical Validation**: paired t-test for significance testing across datasets


## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Quick Start - Local Development

1. **Clone the repository:**
```bash
git clone https://github.com/Seid-Com/FPSO-DBSCAN.git
cd FPSO-DBSCAN
```

2. **Install dependencies:**
```bash
pip install streamlit pandas numpy plotly folium streamlit-folium scikit-learn scipy
```

3. **Run the application:**
```bash
# Option 1: Using the launcher script
python run.py

# Option 2: Direct Streamlit command
streamlit run app.py

# Option 3: Using setup.py installation
pip install -e .
streamlit run app.py
```

4. **Open in browser:**
The application will be available at `http://localhost:8501`

### Alternative Setup Methods

#### Using Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install streamlit pandas numpy plotly folium streamlit-folium scikit-learn scipy

# Run application
streamlit run app.py
```

#### Using Docker
```bash
# Build and run with Docker
docker build -t adaptive-fuzzy-pso-dbscan .
docker run -p 8501:8501 adaptive-fuzzy-pso-dbscan

# Or use Docker Compose
docker-compose up
```

### Troubleshooting Local Setup

- **Port already in use**: Change port with `streamlit run app.py --server.port 8502`
- **Module not found**: Ensure all dependencies are installed with `pip install -r requirements.txt`
- **Permission denied**: Make sure `run.py` is executable with `chmod +x run.py`

## Usage

### Data Format

Upload a CSV file with the following required columns:
- `facility_type`: Categorical (education, health, entertainment)
- `year_established`: Numerical year
- `latitude`: Geographic coordinate
- `longitude`: Geographic coordinate

### Analysis Steps

1. **Upload Dataset**: Use the sidebar file uploader to select your CSV file
2. **Configure Parameters**: Adjust PSO and Fuzzy parameters using the sliders
3. **Run Analysis**: Click "Run Clustering Analysis" to start the process
4. **View Results**: Examine clustering visualizations, performance metrics, and statistical validation
5. **Export Results**: Download analysis results in CSV format

```


## Project Structure

```
├── app.py                 # Main Streamlit application
├── algorithms/
│   ├── pso.py            # Particle Swarm Optimization implementation
│   └── fuzzy_dbscan.py   # Fuzzy DBSCAN clustering algorithm
├── utils/
│   ├── data_processing.py # Data preprocessing utilities
│   ├── visualization.py   # Plotting and visualization functions
│   └── metrics.py         # Performance evaluation metrics
└── .streamlit/
    └── config.toml        # Streamlit configuration
```

## Performance Evaluation

The system evaluates clustering performance using:
- **Silhouette Score**: Primary metric for cluster quality
- **Calinski-Harabasz Score**: Inter-cluster to intra-cluster ratio
- **Davies-Bouldin Score**: Average similarity between clusters
- **Statistical Validation**: Paired t-test for significance testing

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{abdu2026fpso,
  title={Fuzzy--PSO Optimized DBSCAN for Non-Spatial Clustering},
  author={Mehammed, Seid},
  journal={Under review},
  year={2026}
}
```

## Contact

- **Author**: Seid Mehammed Abdu
- **Email**: seid.m@wldu.edu.et
- **Institution**: Department of Computer Science, Woldia University, Ethiopia

## Acknowledgments

- Research methodology based on the manuscript "Fuzzy–PSO Optimized DBSCAN for Non-Spatial Clustering"
- Built using Streamlit, scikit-learn, and Plotly
