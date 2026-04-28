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

df_accuracy = shot_accuracy_by_fields(df, ['SHOT_ZONE_RANGE','ANGLE_SECTOR']).reset_index()

g = sns.FacetGrid(df_accuracy, col='SHOT_ZONE_RANGE', col_wrap=5, height=3)

g.map_dataframe(
    sns.barplot,
    x="ANGLE_SECTOR",
    y=TARGET_VARIABLE
)


plt.show()
