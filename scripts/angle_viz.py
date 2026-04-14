# 3rd party libraries
import matplotlib.pyplot as plt
import seaborn as sns

# own code
from app.config import TARGET_VARIABLE
from app.data_providers import filtered_shots_dataframe, get_shots_dataframe, main_dataframe
from processing.preprocessing import shot_accuracy_by_fields, add_shot_main_action_type_column

# main
df = main_dataframe()

df_accuracy = shot_accuracy_by_fields(df, ['Shot Zone Range','Angle Sector']).reset_index()

g = sns.FacetGrid(df_accuracy, col='Shot Zone Range', col_wrap=4, height=3)

g.map_dataframe(
    sns.barplot,
    x="Angle Sector",
    y=TARGET_VARIABLE
)


plt.show()
