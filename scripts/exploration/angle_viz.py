# 3rd party libraries
import matplotlib.pyplot as plt
import seaborn as sns

# own code
from app.config import TARGET_VARIABLE
from app.data_providers import main_dataframe
from processing.helpers import shot_accuracy_by_fields


# main
df = main_dataframe()

df_accuracy = shot_accuracy_by_fields(df, ['SHOT_ZONE_RANGE','ANGLE_SECTOR']).reset_index()

g = sns.FacetGrid(df_accuracy, col='SHOT_ZONE_RANGE', col_wrap=5, height=3)

# plot angle sector accuracy by shot zone range
g.map_dataframe(
    sns.barplot,
    x="ANGLE_SECTOR",
    y=TARGET_VARIABLE
)


plt.show()
