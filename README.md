# nba_shot_predictions
Collaborative Cap-Stone Project of our DataScientist training

## installation
### Repo and dependencies
checkout from github
in project dir, setup venv and activate. 
Then:
`pip install -r requirements.txt`
to install the required packages and
`pip install -e .`
to install the src package as package in the venv.
### CSV Data
We stopped working with csv data directly and now host our raw data on Huggingface in 
parquet format. Accessing the raw data can be done via the functions in app/data_providers.py.

## scripts

Always run from repo root. For imports to work properly, use "module" type of calling: 
(.venv) nba_shot_predictions$ python -m scripts.scriptname

### Important Scripts
#### Train all classic ML models and persist them
(.venv) nba_shot_predictions$ python -m scripts.modeling.build_persisting_models
#### Train all classic ML models, perform evaluations metrics and log them to doc/results/experiments/model_comparison 
(.venv) nba_shot_predictions$ python -m scripts.prediction.experiments.model_comparison

#### Define our player choice
(.venv) nba_shot_predictions$ python -m scripts.tools.player_choice



