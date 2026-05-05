from huggingface_hub.utils import paginate

from app.conf.run import build_default_run_config, MODEL_ID_LIGHT_GBM
from app.experiments import experiment_current_path, load_runs_to_dataframe
import streamlit as st

from evaluation.insights import get_false_predictions


def sl_show_experiment_results(experiment_id: str):
    df = load_runs_to_dataframe(experiment_current_path(experiment_id))
    st.title("Experiment Dashboard for " + experiment_id)
    default_fields = ['model', 'context_name', 'metric_accuracy', 'macro_avg_precision', 'macro_avg_recall', 'macro_avg_f1-score' ]
    st.dataframe(df[default_fields])

def sl_show_false_predictions():
    config = build_default_run_config()
    config.model_config.model_id = MODEL_ID_LIGHT_GBM
    page_size = 20
    false_positives, false_negatives = get_false_predictions(config)
    total_pages = (len(false_positives) - 1) // page_size + 1
    page_p = st.number_input("Seite", min_value=1, max_value=total_pages, step=1)
    st.title("False Positives")
    st.dataframe(paginate_dataframe(false_positives, page_size, page_p))

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