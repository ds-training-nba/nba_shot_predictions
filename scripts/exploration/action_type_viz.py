# 3rd party libraries
import matplotlib.pyplot as plt
import seaborn as sns

# own code
from app.config import TARGET_VARIABLE
from app.data_providers import main_dataframe
from processing.helpers import shot_accuracy_by_fields

# main
df = main_dataframe()

# get accuracy per Shot technique and player
df_accuracy = shot_accuracy_by_fields(df, ['MAIN_ACTION_TYPE', 'PLAYER_NAME']).reset_index()

############### Players and Main Action Type accuracy
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




# Action Type to Main Action Type mapping
# g = sns.FacetGrid(df, col='MAIN_ACTION_TYPE', col_wrap=2, height=3, sharey=False, sharex=False)
#
# g.map_dataframe(
#     sns.countplot,
#     x="ACTION_TYPE"
# )

############### Players and Main Action Type distribution
g = sns.FacetGrid(df, col='PLAYER_NAME', col_wrap=4, height=3)

g.map_dataframe(
    sns.histplot,
    x="MAIN_ACTION_TYPE",
    stat="probability"
)
g.set_xticklabels(rotation=90)
plt.xticks(rotation=90, ha='right')
plt.show()
