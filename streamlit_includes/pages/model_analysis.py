import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from streamlit_includes.data.load_models import load_metrics, load_splits, load_feature_importance


def render():

    st.markdown(
        '<div class="section-title">📉 Model Evaluation & Analysis</div>',
        unsafe_allow_html=True
    )

    metrics   = load_metrics()
    splits    = load_splits()
    fi_df     = load_feature_importance()

    results_df = metrics["results_df"]
    roc_data   = metrics["roc_data"]
    cm_data    = metrics["cm_data"]
    X_test     = splits["X_test"]
    y_test     = splits["y_test"]

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "ROC-AUC",
        "Confusion Matrix",
        "Metrics",
        "Feature Importance",
        "Comparison",
    ])

    # =========================
    # TAB 1 - ROC
    # =========================
    with tab1:
        st.markdown("### ROC-AUC Curves (All Models)")

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            for model_name, data in roc_data.items():
                auc = results_df.loc[results_df["Model"] == model_name, "ROC-AUC"].values[0]
                fig.add_trace(go.Scatter(
                    x=data["fpr"],
                    y=data["tpr"],
                    mode="lines",
                    name=f"{model_name} (AUC={auc})"
                ))

            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="gray"),
                name="Random (AUC=0.5)"
            ))

            fig.update_layout(
                height=500, width=800,
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=False)

        with col2:
            st.markdown("""
            **Interpretation:**
            - Higher curve = better model
            - XGBoost should dominate in most cases
            - AUC closer to 1 = stronger discrimination ability
            """)

    # =========================
    # TAB 2 - CONFUSION MATRIX
    # =========================
    with tab2:
        st.markdown("### Confusion Matrices — All Models")

        cols = st.columns(len(cm_data))
        for col, (model_name, cm) in zip(cols, cm_data.items()):
            tn, fp, fn, tp = cm.ravel()
            with col:
                st.markdown(f"#### {model_name}")

                # -------------------------
                # HEATMAP
                # -------------------------
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=["Miss", "Made"],
                    y=["Miss", "Made"],
                    colorscale="Blues",
                    text=cm,
                    texttemplate="%{text}",
                    showscale=False
                ))
                fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

                accuracy  = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp + 1e-9)
                recall    = tp / (tp + fn + 1e-9)
                f1        = 2 * precision * recall / (precision + recall + 1e-9)

                st.metric("Accuracy",  f"{accuracy:.3f}")
                st.metric("Precision", f"{precision:.3f}")
                st.metric("Recall",    f"{recall:.3f}")
                st.metric("F1",        f"{f1:.3f}")

    # =========================
    # TAB 3 - METRICS
    # =========================
    with tab3:
        st.markdown("### Model Performance Metrics")

        st.dataframe(
            results_df.sort_values("ROC-AUC", ascending=False),
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(
            results_df.sort_values("ROC-AUC", ascending=True),
            x="ROC-AUC", y="Model",
            orientation="h",
            color="ROC-AUC",
            color_continuous_scale="Viridis",
            title="ROC-AUC by Model"
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # =========================
    # TAB 4 - FEATURE IMPORTANCE
    # =========================
    with tab4:
        st.markdown("### XGBoost — Feature Importance (Top 20)")

        top20 = fi_df.head(20)

        fig = px.bar(
            top20.sort_values("importance"),
            x="importance", y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Oranges",
            title="Top 20 Most Influential Features"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Full feature importance table"):
            st.dataframe(fi_df, use_container_width=True, hide_index=True)

    # =========================
    # TAB 5 - COMPARISON
    # =========================
    with tab5:
        st.markdown("### Model Comparison Overview")

        fig = px.bar(
            results_df,
            x="Model",
            y=["ROC-AUC", "Accuracy", "F1"],
            barmode="group",
            title="Key Metrics — All Models"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        ### Key Insights
        - **XGBoost** performs best on structured NBA shot data
        - **Random Forest** is a strong ensemble baseline
        - **Logistic Regression** gives interpretability at lower accuracy
        - **KNN** struggles with high-dimensional one-hot encoded features
        """)
