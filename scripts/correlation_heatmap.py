import matplotlib.pyplot as plt
import seaborn as sns

from app.data_providers import filtered_shots_dataframe
from app.config import EXPLANATORY_CANDIDATES_NUMERICAL, TARGET_VARIABLE

colums = EXPLANATORY_CANDIDATES_NUMERICAL + [TARGET_VARIABLE]
df_player_choice = filtered_shots_dataframe()[colums]
# g = sns.heatmap(df_player_choice.corr())
#
# plt.show()

import plotly.express as px

fig = px.imshow(df_player_choice.corr())
fig.show()