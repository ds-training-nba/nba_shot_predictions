import streamlit as st
import pandas as pd
import numpy as np
from datasets import load_dataset,concatenate_datasets

from streamlit_includes.data.top_players import load_top_20_players

@st.cache_data
def get_top_20_shots():
    """Load data direct from huggingface"""

    ds = load_dataset(
        "parquet",
        data_files="https://huggingface.co/datasets/ds-training-nba/nba_shot_data/resolve/main/processed/processed_20_players.parquet"
    )

    return ds['train'].to_pandas()