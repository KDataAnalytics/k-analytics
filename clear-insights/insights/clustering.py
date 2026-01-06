import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from ui.icons import heading_html
from ui.navigation import back_button
from utils.plotting import plot_elbow


# ------------------------------------------------------------
# STEP 3D - CLUSTERING ANALYSIS (Fixed Continue Button)
# ------------------------------------------------------------
def run_clustering():
    df = st.session_state.df.copy()
    st.markdown(heading_html("Customer Segmentation (Clustering)", "cluster", level=1), unsafe_allow_html=True)

    st.markdown("Discover natural groups in your data based on similar characteristics.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Clustering requires at least 2 numeric columns.")
        back_button(2)
        return

    st.info(f"Using {len(numeric_cols)} numeric features: {', '.join(numeric_cols)}")

    selected_features = st.multiselect(
        "Select features for clustering (optional - default: all)",
        options=numeric_cols,
        default=numeric_cols
    )

    if not selected_features:
        selected_features = numeric_cols

    X = df[selected_features].copy()
    X = X.dropna()

    if len(X) < 10:
        st.error("Not enough complete rows for clustering (need >=10).")
        return

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.markdown(heading_html("Finding Optimal Number of Clusters", "cluster", level=3), unsafe_allow_html=True)
    with st.spinner("Running elbow analysis..."):
        inertias = []
        K = range(2, min(10, len(X)))
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)

        fig_elbow = plot_elbow(K, inertias)
        st.pyplot(fig_elbow)

    st.info(
        "Use the elbow chart to pick k where the curve starts to flatten. "
        "This suggests diminishing returns from adding more clusters."
    )

    suggested_k = 4
    if len(inertias) >= 3:
        second_diff = np.diff(inertias, 2)
        elbow_idx = int(np.argmax(-second_diff)) + 2
        suggested_k = int(K[elbow_idx]) if elbow_idx < len(inertias) else int(K[-1])
        suggested_k = max(3, min(5, suggested_k))

    st.caption(f"Suggested k based on the elbow: {suggested_k}")
    optimal_k = st.slider("Select number of clusters", 2, 8, suggested_k)

    compare_models = st.checkbox(
        "Compare models (KMeans, Agglomerative, Gaussian Mixture)",
        value=False
    )

    if st.button("Run Clustering", type="primary"):
        with st.spinner(f"Clustering into {optimal_k} groups..."):
            best_model_name = "KMeans"
            clusters = None
            leaderboard = None

            results = []

            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(X_scaled)
            clusters = kmeans_labels

            if compare_models:
                if len(set(kmeans_labels)) > 1:
                    results.append({
                        "Model": "KMeans",
                        "Silhouette": float(silhouette_score(X_scaled, kmeans_labels))
                    })

                agg = AgglomerativeClustering(n_clusters=optimal_k, linkage="ward")
                agg_labels = agg.fit_predict(X_scaled)
                if len(set(agg_labels)) > 1:
                    results.append({
                        "Model": "Agglomerative",
                        "Silhouette": float(silhouette_score(X_scaled, agg_labels))
                    })

                gmm = GaussianMixture(n_components=optimal_k, random_state=42)
                gmm_labels = gmm.fit_predict(X_scaled)
                if len(set(gmm_labels)) > 1:
                    results.append({
                        "Model": "Gaussian Mixture",
                        "Silhouette": float(silhouette_score(X_scaled, gmm_labels))
                    })

                if results:
                    leaderboard = (
                        pd.DataFrame(results)
                        .sort_values("Silhouette", ascending=False)
                        .reset_index(drop=True)
                    )
                    st.markdown(heading_html("Model comparison", "chart", level=3), unsafe_allow_html=True)
                    st.dataframe(
                        leaderboard.style.format({"Silhouette": "{:.3f}"}),
                        use_container_width=True
                    )

                    best_model_name = leaderboard.iloc[0]["Model"]
                    if best_model_name == "Agglomerative":
                        clusters = agg_labels
                    elif best_model_name == "Gaussian Mixture":
                        clusters = gmm_labels

            if clusters is None:
                st.error("Clustering failed. Please try again.")
                return

            df_clustered = df.loc[X.index].copy()
            df_clustered["Cluster"] = clusters

            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            # Fixed scatter plot
            fig_scatter = px.scatter(
                df_clustered,
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                color="Cluster",
                hover_data=selected_features,
                title="Customer Segments (PCA Projection)",
                labels={"x": "PCA Component 1", "y": "PCA Component 2"}
            )
            fig_scatter.update_traces(marker=dict(size=12, opacity=0.8))
            st.plotly_chart(fig_scatter, use_container_width=True)

            with st.expander("How to read the PCA chart", expanded=False):
                st.write(
                    "This chart projects your multi-feature data into 2D so you can see cluster separation. "
                    "Each point is a customer; colors show cluster membership. "
                    "Tighter groups mean more similarity, and more distance between groups means clearer separation. "
                    "The axes are principal components, so the exact numbers are less important than the relative spacing."
                )

            # Profiles
            st.markdown(heading_html("Cluster Profiles", "cluster", level=3), unsafe_allow_html=True)
            profile = df_clustered.groupby("Cluster")[selected_features].mean().round(2)
            profile = profile.T
            profile.columns = [f"Cluster {i}" for i in profile.columns]
            st.dataframe(profile.style.background_gradient(cmap="viridis"), use_container_width=True)

            # Sizes
            st.markdown(heading_html("Cluster Sizes", "chart", level=3), unsafe_allow_html=True)
            sizes = df_clustered["Cluster"].value_counts().sort_index()
            fig_sizes = px.bar(x=[f"Cluster {i}" for i in sizes.index], y=sizes.values,
                               labels={"x": "Cluster", "y": "Customers"},
                               title="Customers per Cluster")
            st.plotly_chart(fig_sizes, use_container_width=True)

            # Export
            st.markdown(heading_html("Export Segmented Data", "download", level=3), unsafe_allow_html=True)
            csv = df_clustered.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Data with Cluster Labels",
                data=csv,
                file_name="customer_segments_with_clusters.csv",
                mime="text/csv",
                use_container_width=True
            )

            # Store results
            st.session_state.clustering_results = {
                "df_clustered": df_clustered,
                "features": selected_features,
                "n_clusters": optimal_k,
                "profile": profile,
                "best_model_name": best_model_name,
                "leaderboard": leaderboard
            }

    # === Continue button - always visible after clustering ===
    st.markdown("---")
    back_button(2)

    if 'clustering_results' in st.session_state:
        if st.button("Continue -> Insights Summary", type="primary", key="clustering_continue", use_container_width=True):
            st.session_state.current_analysis = "clustering"
            st.session_state.step = 5
            st.rerun()
    else:
        st.info("Run clustering above to continue.")

    st.session_state.analysis_complete = True
