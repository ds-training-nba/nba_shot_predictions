from app.config import TARGET_VARIABLE
from app.data_providers import main_dataframe


# main
df = main_dataframe()

from scipy.stats import pearsonr
print(pearsonr(df['SHOT_DISTANCE'],df[TARGET_VARIABLE]))
print(pearsonr(df['PERIOD_x'],df[TARGET_VARIABLE]))