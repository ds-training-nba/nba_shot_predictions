# 3rd party libraries
import matplotlib.pyplot as plt
import seaborn as sns

# own code
from app.config import TARGET_VARIABLE
from app.data_providers import main_dataframe
from processing.helpers import shot_accuracy_by_fields

# raw data
df = main_dataframe()

df_accuracy = shot_accuracy_by_fields(df, ['SHOT_ZONE_RANGE', 'PLAYER_NAME']).reset_index()
# shot accuracy distribution by player and Range
g = sns.FacetGrid(df_accuracy, col="SHOT_ZONE_RANGE", col_wrap=4, height=3)

# g.map_dataframe(
#     sns.barplot,
#     x='PLAYER_NAME',
#     y=TARGET_VARIABLE
# )

#
g.map_dataframe(
    sns.boxplot,
    y=TARGET_VARIABLE
)
plt.show()
