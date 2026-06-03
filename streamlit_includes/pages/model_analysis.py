import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from streamlit_includes.data.top_players import load_top_20_players
from streamlit_includes.data.top_20_dataset import get_top_20_shots
from streamlit_includes.data.train_data import train_models


def render():

    df = get_top_20_shots()

    st.markdown(
        '<div class="section-title">📉 Model Evaluation & Analysis</div>',
        unsafe_allow_html=True
    )

    # =========================
    # TRAIN (cached)
    # =========================
    with st.spinner("Training ML models..."):
        models, results_df, roc_data, cm_data, X_train, X_test, y_train, y_test = train_models(df)

    # =========================
    # TABS
    # =========================
    tab1, tab2, tab3, tab4 = st.tabs([
        "ROC-AUC",
        "Confusion Matrix",
        "Metrics",
        "Comparison"
    ])

    # =========================
    # TAB 1 - ROC
    # =========================
    with tab1:
        st.markdown("### ROC-AUC Curves (All Models)")

        fig = go.Figure()

        for model_name, data in roc_data.items():
            fig.add_trace(go.Scatter(
                x=data["fpr"],
                y=data["tpr"],
                mode="lines",
                name=model_name
            ))

        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Random"
        ))

        fig.update_layout(
            height=500,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

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
        st.markdown("### Confusion Matrices - All Models Comparison")

        cols = st.columns(len(cm_data))  # 4 модели → 4 колонки

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

                fig.update_layout(
                    height=250,
                    margin=dict(l=10, r=10, t=30, b=10)
                )

                st.plotly_chart(fig, use_container_width=True)

                # -------------------------
                # METRICS
                # -------------------------
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp + 1e-9)
                recall = tp / (tp + fn + 1e-9)

                st.metric("Acc", f"{accuracy:.2f}")
                st.metric("Prec", f"{precision:.2f}")
                st.metric("Rec", f"{recall:.2f}")

                st.markdown(
                    f"""
                    **TP:** {tp}  
                    **TN:** {tn}  
                    **FP:** {fp}  
                    **FN:** {fn}
                    """
                )

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
            results_df,
            x="Model",
            y="ROC-AUC",
            color="ROC-AUC",
            color_continuous_scale="Viridis"
        )

        fig.update_layout(height=450)

        st.plotly_chart(fig, use_container_width=True)

    # =========================
    # TAB 4 - COMPARISON
    # =========================
    with tab4:
        st.markdown("### Model Comparison Overview")

        fig = px.bar(
            results_df,
            x="Model",
            y=["ROC-AUC", "Accuracy", "Log Loss"],
            barmode="group"
        )

        fig.update_layout(height=500)

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        ### Key Insights
        - XGBoost typically performs best in structured NBA shot data
        - Random Forest is strong baseline
        - Logistic Regression gives interpretability
        - KNN struggles with high-dimensional features
        """)
