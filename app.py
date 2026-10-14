import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium
import io
import time
from scipy import stats
from sklearn.cluster import DBSCAN

from algorithms.pso import PSO
from algorithms.fuzzy_dbscan import FuzzyDBSCAN
from utils.data_processing import preprocess_data, z_score_normalize
from utils.visualization import create_feature_visualization, create_parameter_evolution_plot
from utils.metrics import calculate_additional_metrics

# Configure page
st.set_page_config(
    page_title="Fuzzy-PSO Optimized DBSCAN",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Non-spatial attributes used for clustering (manuscript, Figure 3):
# x = normalized year of establishment, y = numeric service type.
CLUSTER_FEATURES = ['year_established', 'facility_type_encoded']

def main():
    st.title("🏙️ Fuzzy–PSO Optimized DBSCAN for Non-Spatial Clustering")
    st.markdown("""
    Implementation of the hybrid fuzzy–particle swarm optimization DBSCAN framework
    by **Seid Mehammed Abdu** (Woldia University, Ethiopia). PSO is embedded in the
    fuzzy clustering loop and continuously optimizes **ε, MinPtsMin and MinPtsMax**
    using the Silhouette coefficient as fitness, while fuzzy membership handles
    uncertainty of points near cluster boundaries. Clustering runs on normalized
    **non-spatial** attributes; GPS coordinates are used only for GIS display.
    """)

    # Sidebar for configuration
    st.sidebar.title("Configuration")

    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV Dataset",
        type=['csv'],
        help="Upload a CSV file with columns: facility_type, year_established, latitude, longitude"
    )

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Dataset loaded: {len(data)} records")

            required_columns = ['facility_type', 'year_established', 'latitude', 'longitude']
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                st.stop()

            # Display data overview
            st.subheader("📊 Dataset Overview")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Records", len(data))
            with col2:
                st.metric("Service Types", data['facility_type'].nunique())
            with col3:
                st.metric("Time Range", f"{int(data['year_established'].min())}-{int(data['year_established'].max())}")

            with st.expander("View Data Sample", expanded=False):
                st.dataframe(data.head(10))

            # PSO Parameters (Table 3 of the manuscript)
            st.sidebar.subheader("PSO Parameters (Table 3)")
            n_particles = st.sidebar.slider("Swarm size (particles)", 10, 100, 30)
            n_iterations = st.sidebar.slider("Maximum iterations", 20, 300, 100)
            w = st.sidebar.slider("Inertia weight (w)", 0.1, 1.0, 0.7)
            c1 = st.sidebar.slider("Cognitive coefficient (c₁)", 0.1, 3.0, 2.0)
            c2 = st.sidebar.slider("Social coefficient (c₂)", 0.1, 3.0, 2.0)

            st.sidebar.caption(
                "ε search range: 0.10–1.00 · MinPtsMin: 2–10 · MinPtsMax: 5–20 · "
                "fitness: Silhouette coefficient"
            )

            # Temporal analysis option (Table 2: before / after 2011)
            temporal_analysis = st.sidebar.checkbox(
                "Temporal analysis (before / from 2011)", value=False)

            if st.sidebar.button("🚀 Run Clustering Analysis", type="primary"):
                run_clustering_analysis(data, n_particles, n_iterations, w, c1, c2,
                                        temporal_analysis)

        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
    else:
        st.info("👆 Please upload a CSV dataset to begin analysis")

        st.subheader("📋 Expected Data Format")
        example_data = pd.DataFrame({
            'facility_type': ['education', 'health', 'entertainment', 'education'],
            'year_established': [2005, 2012, 2008, 2015],
            'latitude': [9.0307, 9.0397, 9.0257, 9.0407],
            'longitude': [38.7407, 38.7507, 38.7357, 38.7557]
        })
        st.dataframe(example_data)

        st.markdown("""
        **Required Columns:**
        - `facility_type`: Categorical service type (education, health, entertainment, ...)
        - `year_established`: Numerical year
        - `latitude` / `longitude`: used only for the GIS map (not for clustering)

        Clustering uses the normalized non-spatial attributes
        (`year_established`, encoded `facility_type`), as described in the manuscript.
        """)


def prepare_features(data):
    """Z-score normalize the non-spatial attributes used for clustering."""
    return z_score_normalize(data[CLUSTER_FEATURES])


def run_clustering_analysis(data, n_particles, n_iterations, w, c1, c2,
                            temporal_analysis):
    """Execute the complete clustering analysis pipeline (Algorithm 2)."""

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Step 1: Data preprocessing + z-score normalization (equation 1)
        status_text.text("Step 1/6: Preprocessing and z-score normalization...")
        progress_bar.progress(10)

        processed_data = preprocess_data(data)

        if temporal_analysis:
            datasets = {
                'Sectors founded before 2011':
                    processed_data[processed_data['year_established'] < 2011],
                'Sectors founded from 2011':
                    processed_data[processed_data['year_established'] >= 2011],
            }
        else:
            datasets = {'Full Dataset': processed_data}

        all_results = {}

        for dataset_name, orig_data in datasets.items():
            if len(orig_data) < 10:
                st.warning(f"Skipping '{dataset_name}': fewer than 10 records.")
                continue

            norm_data = prepare_features(orig_data)

            st.subheader(f"🔬 Analysis Results: {dataset_name}")

            # Step 2: Proposed method — PSO embedded in the fuzzy loop
            status_text.text(f"Step 2/6: Fuzzy–PSO optimization for {dataset_name}...")
            progress_bar.progress(25)

            pso = PSO(n_particles=n_particles, n_iterations=n_iterations,
                      w=w, c1=c1, c2=c2, mode='fuzzy')
            t0 = time.perf_counter()
            best = pso.optimize(norm_data)
            proposed_time = time.perf_counter() - t0

            proposed_model = FuzzyDBSCAN(eps=best['eps'],
                                         min_pts_min=best['min_pts_min'],
                                         min_pts_max=best['min_pts_max'])
            proposed_labels, proposed_memberships = proposed_model.fit_predict(norm_data)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Optimized ε", f"{best['eps']:.4f}")
            with col2:
                st.metric("MinPtsMin", best['min_pts_min'])
            with col3:
                st.metric("MinPtsMax", best['min_pts_max'])
            with col4:
                st.metric("Best fitness (Silhouette)", f"{best['fitness']:.4f}")

            # Step 3: Baselines for comparison (Table 4)
            status_text.text(f"Step 3/6: Running baseline methods for {dataset_name}...")
            progress_bar.progress(45)

            # (a) Classical DBSCAN
            t0 = time.perf_counter()
            dbscan_labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(norm_data)
            dbscan_time = time.perf_counter() - t0

            # (b) Fuzzy DBSCAN without parameter optimization
            t0 = time.perf_counter()
            fuzzy_plain_model = FuzzyDBSCAN(eps=0.5, min_pts_min=3, min_pts_max=10)
            fuzzy_plain_labels, _ = fuzzy_plain_model.fit_predict(norm_data)
            fuzzy_plain_time = time.perf_counter() - t0

            # (c) PSO-DBSCAN — PSO over the classic two-parameter DBSCAN
            status_text.text(f"Step 4/6: PSO-DBSCAN baseline for {dataset_name}...")
            progress_bar.progress(60)

            pso_base = PSO(n_particles=n_particles, n_iterations=n_iterations,
                           w=w, c1=c1, c2=c2, mode='dbscan')
            t0 = time.perf_counter()
            base_best = pso_base.optimize(norm_data)
            pso_dbscan_time = time.perf_counter() - t0
            pso_dbscan_labels = DBSCAN(eps=base_best['eps'],
                                       min_samples=base_best['min_pts']
                                       ).fit_predict(norm_data)

            # Step 4: Evaluate all methods with internal validation indices
            status_text.text(f"Step 5/6: Evaluating clustering quality for {dataset_name}...")
            progress_bar.progress(75)

            methods = {
                'DBSCAN': (dbscan_labels, dbscan_time),
                'Fuzzy DBSCAN': (fuzzy_plain_labels, fuzzy_plain_time),
                'PSO-DBSCAN': (pso_dbscan_labels, pso_dbscan_time),
                'Proposed Fuzzy–PSO': (proposed_labels, proposed_time),
            }

            comparison_rows = []
            for method_name, (labels, exec_time) in methods.items():
                extra = calculate_additional_metrics(norm_data, labels)
                comparison_rows.append({
                    'Method': method_name,
                    'Silhouette': extra['silhouette_score'],
                    'Davies-Bouldin ↓': extra['davies_bouldin_score'],
                    'Calinski-Harabasz ↑': extra['calinski_harabasz_score'],
                    'Execution Time (s)': round(exec_time, 3),
                    'Clusters': extra['n_clusters'],
                    'Noise points': extra['n_noise_points'],
                })

            comparison_df = pd.DataFrame(comparison_rows)
            st.markdown(f"**Method comparison — {dataset_name}**")
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            proposed_sil = comparison_df.loc[
                comparison_df['Method'] == 'Proposed Fuzzy–PSO', 'Silhouette'].iloc[0]
            dbscan_sil = comparison_df.loc[
                comparison_df['Method'] == 'DBSCAN', 'Silhouette'].iloc[0]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Proposed Silhouette", f"{proposed_sil:.4f}")
            with col2:
                if dbscan_sil > 0:
                    improvement = ((proposed_sil - dbscan_sil) / dbscan_sil) * 100
                    st.metric("Improvement over DBSCAN", f"{improvement:.1f}%",
                              delta=f"{proposed_sil - dbscan_sil:.4f}")
                else:
                    st.metric("Improvement over DBSCAN", "n/a")

            all_results[dataset_name] = {
                'comparison': comparison_df,
                'proposed_silhouette': proposed_sil,
                'dbscan_silhouette': dbscan_sil,
                'labels': {name: labels for name, (labels, _) in methods.items()},
                'memberships': proposed_memberships,
                'best_params': best,
                'data': orig_data,
                'normalized_data': norm_data,
            }

            # Step 5: Visualizations
            status_text.text(f"Step 6/6: Creating visualizations for {dataset_name}...")
            progress_bar.progress(90)

            create_visualizations(orig_data, norm_data, methods,
                                  proposed_memberships, dataset_name)

            if len(best.get('history', [])) > 1:
                st.subheader(f"📈 PSO Convergence — {dataset_name}")
                fig_conv = create_parameter_evolution_plot(best['history'])
                if fig_conv is not None:
                    st.plotly_chart(fig_conv, use_container_width=True)

        if len(all_results) > 1:
            perform_statistical_validation(all_results)

        create_export_section(all_results)

        progress_bar.progress(100)
        status_text.text("✅ Analysis completed successfully!")

    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        st.exception(e)


def create_visualizations(data, norm_data, methods, memberships, dataset_name):
    """Create cluster comparisons, membership analysis and GIS map."""

    # Non-spatial clustering views (year of establishment vs service type)
    st.subheader(f"🧭 Cluster Comparison (non-spatial features) — {dataset_name}")
    cols = st.columns(len(methods))
    for col, (method_name, (labels, _)) in zip(cols, methods.items()):
        with col:
            fig = create_feature_visualization(norm_data, labels, method_name)
            st.plotly_chart(fig, use_container_width=True)

    # Fuzzy membership distribution of the proposed method
    st.subheader(f"🔍 Fuzzy Membership Analysis — {dataset_name}")
    clustered = memberships > 0
    if np.any(clustered):
        membership_df = pd.DataFrame({'Membership': memberships[clustered]})
        fig_hist = px.histogram(
            membership_df, x='Membership', nbins=30,
            title="Distribution of fuzzy membership degrees "
                  "(points near boundaries have intermediate values)",
            color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig_hist, use_container_width=True)

    # Interactive GIS map (GPS coordinates used for display only)
    st.subheader(f"🗺️ GIS Visualization — {dataset_name}")
    proposed_labels = methods['Proposed Fuzzy–PSO'][0]

    center_lat = data['latitude'].mean()
    center_lon = data['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
              'lightred', 'beige', 'darkblue', 'darkgreen']

    for i in range(len(data)):
        row = data.iloc[i]
        label = proposed_labels[i]
        if label == -1:
            color = 'black'
            popup_text = (f"Noise<br>Type: {row['facility_type']}"
                          f"<br>Year: {row['year_established']}")
        else:
            color = colors[int(label) % len(colors)]
            popup_text = (f"Cluster {label}<br>Type: {row['facility_type']}"
                          f"<br>Year: {row['year_established']}"
                          f"<br>Membership: {memberships[i]:.3f}")

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=6,
            popup=popup_text,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)

    st_folium(m, width=700, height=500)


def perform_statistical_validation(results):
    """Paired t-test of the proposed method against classical DBSCAN."""
    st.subheader("📈 Statistical Validation")

    proposed = [r['proposed_silhouette'] for r in results.values()]
    baseline = [r['dbscan_silhouette'] for r in results.values()]

    if len(proposed) >= 2:
        t_stat, p_value = stats.ttest_rel(proposed, baseline)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("T-statistic", f"{t_stat:.4f}")
        with col2:
            st.metric("P-value", f"{p_value:.6f}")
        with col3:
            st.metric("Result", "Significant" if p_value < 0.05 else "Not Significant")

        if p_value < 0.05:
            st.success("✅ The improvement is statistically significant (p < 0.05)")
        else:
            st.warning("⚠️ The improvement is not statistically significant (p ≥ 0.05)")

        comparison_df = pd.concat(
            [r['comparison'] for r in results.values()], ignore_index=True)
        comparison_df['Dataset'] = list(results.keys()) * len(comparison_df) \
            if len(comparison_df) == len(results) else ''

        fig_comparison = px.bar(
            comparison_df,
            x='Dataset',
            y='Silhouette',
            color='Method',
            barmode='group',
            title='Silhouette Comparison Across Datasets and Methods'
        )
        st.plotly_chart(fig_comparison, use_container_width=True)


def create_export_section(results):
    """Export summary of all methods and datasets."""
    st.subheader("📥 Export Results")

    summary_frames = []
    for dataset_name, result in results.items():
        frame = result['comparison'].copy()
        frame.insert(0, 'Dataset', dataset_name)
        summary_frames.append(frame)

    if not summary_frames:
        return

    summary_df = pd.concat(summary_frames, ignore_index=True)

    csv_buffer = io.StringIO()
    summary_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📊 Download Results CSV",
        data=csv_buffer.getvalue(),
        file_name="fuzzy_pso_dbscan_results.csv",
        mime="text/csv"
    )

    if st.button("📋 Show Detailed Results"):
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
