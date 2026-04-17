from app.config import TARGET_VARIABLE
from app.data_providers import filtered_shots_dataframe, get_shots_dataframe, main_dataframe
from processing.preprocessing import shot_accuracy_by_fields, add_shot_main_action_type_column

# main
df = main_dataframe()

from scipy.stats import pearsonr
#print(pearsonr(df['SHOT_DISTANCE'],df[TARGET_VARIABLE]))
print(pearsonr(df['PERIOD_x'],df[TARGET_VARIABLE]))