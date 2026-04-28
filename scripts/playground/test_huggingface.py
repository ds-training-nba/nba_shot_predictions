from app.data_providers import get_shots_dataframe_from_huggingface
df = get_shots_dataframe_from_huggingface()
print(df.head())