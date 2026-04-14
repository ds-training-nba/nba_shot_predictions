import matplotlib.pyplot as plt
import seaborn as sns

from app.data_providers import filtered_shots_dataframe, main_dataframe
from app.config import EXPLANATORY_CANDIDATES_NUMERICAL, TARGET_VARIABLE

colums = EXPLANATORY_CANDIDATES_NUMERICAL + [TARGET_VARIABLE]
df_player_choice = main_dataframe()
df_player_choice = df_player_choice[df_player_choice['SHOT_ZONE_RANGE'] == "Less Than 8 ft."]
print(df_player_choice.describe())
print(df_player_choice.head(200).tail(30))
df_player_choice = df_player_choice[colums]
# g = sns.heatmap(df_player_choice.corr())
#
# plt.show()

import plotly.express as px


print(df_player_choice.corr())
fig = px.imshow(df_player_choice.corr())
fig.show()