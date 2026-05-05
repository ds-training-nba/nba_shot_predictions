from app.streamlit import sl_show_experiment_results, sl_show_false_predictions
import sys

match(sys.argv[1]):
    case 'experiment_results':
        experiment_id = sys.argv[2]
        sl_show_experiment_results(experiment_id)
    case 'analyze_false_predictions':
        sl_show_false_predictions()