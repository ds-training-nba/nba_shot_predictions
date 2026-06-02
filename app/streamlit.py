import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from app.conf.run import build_best_run_config
from app.experiments import experiment_current_path, load_runs_to_dataframe
from evaluation.insights import get_false_predictions


def sl_show_experiment_results(experiment_id: str):
    """
    builds a streamlit page with the results table of an experiment
    :param experiment_id:
    :return:
    """
    df = load_runs_to_dataframe(experiment_current_path(experiment_id))
    st.title("Experiment Dashboard for " + experiment_id)
    default_fields = ['model', 'context_name', 'metric_accuracy', 'macro_avg_precision',
                      'macro_avg_recall', 'macro_avg_f1-score']
    if 'metric_brier_score' in df.columns:
        default_fields.insert(2,'metric_brier_score')
    if 'brier_decomposition_10_resolution' in df.columns:
        default_fields.insert(2,'brier_decomposition_10_resolution')
    if 'brier_decomposition_10_reliability' in df.columns:
        default_fields.insert(2, 'brier_decomposition_10_reliability')
    if 'brier_decomposition_20_resolution' in df.columns:
        default_fields.insert(2,'brier_decomposition_20_resolution')
    if 'brier_decomposition_20_reliability' in df.columns:
        default_fields.insert(2, 'brier_decomposition_20_reliability')
    st.dataframe(df[default_fields])

def sl_show_false_predictions():
    """
    builds a streamlit page to analyze false predictions

    :return:
    """
    config = build_best_run_config()
    page_size = 20
    false_positives, false_negatives = get_false_predictions(config)
    total_pages = (len(false_positives) - 1) // page_size + 1
    page_p = st.number_input("Seite", min_value=1, max_value=total_pages, step=1)
    st.title("False Positives")
    st.dataframe(paginate_dataframe(false_positives, page_size, page_p))

    plt.figure(figsize=(10, 8))
    sns.countplot(false_positives, x="MAIN_ACTION_TYPE",stat="percent")

    # Show the plot in Streamlit
    st.pyplot(plt)

    total_pages = (len(false_negatives) - 1) // page_size + 1
    page_n = st.number_input("Seite", min_value=1, max_value=total_pages, step=1)
    st.title("False negatives")
    st.dataframe(paginate_dataframe(false_negatives, page_size, page_n))

def paginate_dataframe(dataframe, page_size, page_num):

    page_size = page_size

    if page_size is None:

        return None

    offset = page_size*(page_num-1)

    return dataframe[offset:offset + page_size]