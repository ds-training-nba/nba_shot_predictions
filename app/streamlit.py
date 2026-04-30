from app.experiments import experiment_current_path, load_runs_to_dataframe
import streamlit as st

def sl_show_experiment_results(experiment_id: str):
    df = load_runs_to_dataframe(experiment_current_path(experiment_id))
    st.title("Experiment Dashboard for " + experiment_id)
    default_fields = ['model', 'context_name', 'metric_accuracy', 'macro_avg_precision', 'macro_avg_recall', 'macro_avg_f1-score' ]
    st.dataframe(df[default_fields])