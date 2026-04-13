import matplotlib.pyplot as plt
import seaborn as sns

from app.data_providers import filtered_shots_dataframe

df_player_choice = filtered_shots_dataframe()
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