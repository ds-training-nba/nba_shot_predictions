from app.config import TARGET_VARIABLE
from app.data_providers import full_dataset

df = full_dataset()
print("General Hit Rate", df[TARGET_VARIABLE].value_counts(normalize=True))
df_fg_only =  df[df['points'] != 1]
print("Field Goal Only Hit Rate", df_fg_only[TARGET_VARIABLE].value_counts(normalize=True))