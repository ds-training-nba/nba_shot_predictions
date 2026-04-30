from app.streamlit import sl_show_experiment_results
import sys

experiment_id = sys.argv[1]
sl_show_experiment_results(experiment_id)