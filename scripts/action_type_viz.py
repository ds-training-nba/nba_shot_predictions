# 3rd party libraries
import matplotlib.pyplot as plt
import seaborn as sns

# own code
from app.config import TARGET_VARIABLE
from app.data_providers import get_shots_dataframe, main_dataframe, filtered_shots_dataframe
from processing.helpers import shot_accuracy_by_fields
from processing.compute_columns import add_shot_main_action_type_column

# main
df = main_dataframe()

df_accuracy = shot_accuracy_by_fields(df, ['MAIN_ACTION_TYPE', 'PLAYER_NAME']).reset_index()

# g = sns.FacetGrid(df_accuracy, col="MAIN_ACTION_TYPE", col_wrap=4, height=3)
#
# g.map_dataframe(
#     sns.barplot,
#     x='PLAYER_NAME',
#     y=TARGET_VARIABLE
# )

# g = sns.FacetGrid(df_accuracy, col='PLAYER_NAME', col_wrap=4, height=3)
#
# g.map_dataframe(
#     sns.barplot,
#     x="MAIN_ACTION_TYPE",
#     y=TARGET_VARIABLE
# )

g = sns.FacetGrid(df, col='PLAYER_NAME', col_wrap=4, height=3)

g.map_dataframe(
    sns.histplot,
    x="MAIN_ACTION_TYPE",
    stat="probability"
)



plt.show()