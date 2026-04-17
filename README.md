# nba_shot_predictions
Collaborative Cap-Stone Project of our DataScientist training

## installation
### Repo and dependencies
checkout from github
in project dir, setup venv and activate. 
Then:
`pip install -r requirements.txt`

### CSV Data
We stopped working with csv data directly and now host our raw data on Huggingface in 
parquet format. Accessing the raw data can be done via the functions in app/data_providers.py.

## scripts

Always run from repo root. For imports to work properly, use "module" type of calling: 
(.venv) nba_shot_predictions$ python -m scripts.scriptname

### Existing Scripts

(.venv) nba_shot_predictions$ python -m scripts.pandas_test


