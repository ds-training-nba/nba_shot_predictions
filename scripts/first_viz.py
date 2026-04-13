import pandas as pd
import seaborn as sns
from app.config import PLAYER_CHOICE
file = 'data/orig/shots.csv'
df = pd.read_csv(file)

import matplotlib.pyplot as plt

chosen_player_matrix = None

for name in PLAYER_CHOICE:
    current_matrix = df['Player Name'] == name
    if chosen_player_matrix is None:
        chosen_player_matrix = current_matrix
    else:
        chosen_player_matrix = chosen_player_matrix | current_matrix

df_player_choice = df[chosen_player_matrix]

g = sns.FacetGrid(df_player_choice, col="Player Name", col_wrap=4, height=3)

g.map_dataframe(
    sns.kdeplot,
    x="X Location",
    y="Y Location",
    fill=True,
    cmap="magma",
    bw_adjust=0.7
)

plt.show()