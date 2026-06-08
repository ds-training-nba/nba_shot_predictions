import streamlit as st
import pandas as pd
import numpy as np
from datasets import load_dataset,concatenate_datasets

from streamlit_includes.data.top_players import load_top_20_players

@st.cache_data
def get_top_20_shots():
    # ds = load_dataset(
    #     "parquet",
    #     data_files=\"https://huggingface.co/datasets/ds-training-nba/nba_shot_data/resolve/main/raw_merged/merged_dataset.parquet\"\n",
    # )\n",
    # ds = load_dataset(
    #     "parquet",
    #     'ds-training-nba/nba_shot_data',
    #     data_files='data/processed/processed_20_players.parquet'
    # )

    ds = load_dataset("ds-training-nba/nba_shot_data")

    full = concatenate_datasets([
        ds["train"],
        ds["test"]
    ])
    return full.to_pandas()
    # return ds['full'].to_pandas()