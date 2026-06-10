import sys

from app.streamlit import sl_show_experiment_results, sl_show_false_predictions, sl_player_app, sl_alternatives_app

# streamlit app/page chosen via script parameters
match(sys.argv[1]):
    case 'experiment_results':
        experiment_id = sys.argv[2]
        sl_show_experiment_results(experiment_id)
    case 'analyze_false_predictions':
        sl_show_false_predictions()
    case 'player_app':
        sl_player_app()
    case 'alternatives_app':
        sl_alternatives_app()