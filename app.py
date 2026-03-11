import streamlit as st
import pandas as pd

from data_pipeline import load_and_prepare_data
from model_pipeline import (
    build_isolation_forest_pipeline,
    train_model_and_score_anomalies,
    explain_anomalies_in_plain_language,
)


def main():
    st.set_page_config(
        page_title="anomaly detection prototype",
        layout="wide",
    )

    st.title("housing anomaly detection – california housing")
    st.markdown(
        """
this small app walks through a simple anomaly detection workflow:

- **load and clean data** from the california housing dataset  
- **prepare numeric features** with basic preprocessing and scaling  
- **detect unusual areas** using an isolation forest model  
- **explore the results** through tables, charts, and a short text explanation  
        """
    )

    with st.sidebar:
        st.header("Experiment settings")
        sample_size = st.slider(
            "Sample size",
            min_value=5000,
            max_value=30000,
            value=10000,
            step=1000,
            help="How many rows to randomly sample from the dataset.",
        )
        contamination = st.slider(
            "Expected anomaly fraction (contamination)",
            min_value=0.005,
            max_value=0.10,
            value=0.02,
            step=0.005,
            help="Your best guess for what fraction of rows are anomalous.",
        )

    st.subheader("1. Data Loading & Processing")
    with st.spinner("Loading and preparing data..."):
        prepared_data = load_and_prepare_data(sample_size=sample_size)
    st.write(
        f"The app is working with **{len(prepared_data)}** rows and "
        f"**{prepared_data.shape[1]}** numeric features after preprocessing."
    )

    st.markdown("Here is a quick preview of the cleaned data:")
    st.dataframe(prepared_data.head(), use_container_width=True)

    st.subheader("2. Train the model and find anomalies")
    if st.button("Run anomaly detection", type="primary"):
        with st.spinner("Training Isolation Forest and scoring anomalies..."):
            model = build_isolation_forest_pipeline(contamination=contamination)
            scored_data = train_model_and_score_anomalies(model, prepared_data)

        anomaly_count = int(scored_data["is_anomaly"].sum())
        st.success(
            f"The model finished training and flagged **{anomaly_count}** anomalies "
            f"out of **{len(scored_data)}** total rows."
        )

        tab1, tab2, tab3 = st.tabs(
            ["Detected Anomalies", "Visualizations", "AI-style Explanation"]
        )

        with tab1:
            st.markdown("### Detected anomalies")
            anomalies_only = scored_data[scored_data["is_anomaly"] == 1].copy()
            anomalies_sorted = anomalies_only.sort_values(
                by="anomaly_score", ascending=False
            )
            st.write(
                "The table below shows the rows the model considers most unusual, "
                "sorted by highest anomaly score first."
            )
            st.dataframe(
                anomalies_sorted.head(500),
                use_container_width=True,
            )

        with tab2:
            st.markdown("### Anomaly score distribution")
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(
                scored_data["anomaly_score"],
                bins=40,
                kde=True,
                ax=ax,
                color="#1f77b4",
            )
            ax.set_title("How unusual each row looks (anomaly score distribution)")
            ax.set_xlabel("Anomaly score")
            st.pyplot(fig)

            st.markdown("### Which numeric features stand out for anomalies?")
            numeric_feature_cols = [
                c
                for c in scored_data.select_dtypes(include=["number"]).columns
                if c not in ["anomaly_score", "is_anomaly"]
            ]

            anomalies_only = scored_data[scored_data["is_anomaly"] == 1]
            normal_points = scored_data[scored_data["is_anomaly"] == 0]

            if numeric_feature_cols and len(anomalies_only) > 0 and len(normal_points) > 0:
                mean_anom = anomalies_only[numeric_feature_cols].mean()
                mean_norm = normal_points[numeric_feature_cols].mean()
                diff = (mean_anom - mean_norm).abs().sort_values(ascending=False)
                top_features = diff.head(5).index.tolist()

                fig2, ax2 = plt.subplots(figsize=(8, 4))
                bar_data = pd.DataFrame(
                    {
                        "feature": top_features,
                        "anomaly_mean": mean_anom[top_features].values,
                        "normal_mean": mean_norm[top_features].values,
                    }
                ).set_index("feature")
                bar_data.plot(kind="bar", ax=ax2)
                ax2.set_title("Average numeric feature values: anomalies vs normal (top 5)")
                ax2.set_ylabel("Mean (scaled feature value)")
                st.pyplot(fig2)
            else:
                st.write(
                    "The model found anomalies, but there were not enough numeric "
                    "features to build a stable comparison chart."
                )

        with tab3:
            st.markdown("### Short plain-language explanation")
            explanation = explain_anomalies_in_plain_language(scored_data)
            st.write(explanation)


if __name__ == "__main__":
    main()

