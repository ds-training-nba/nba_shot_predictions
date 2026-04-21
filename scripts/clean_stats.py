from app.data_providers import clean_source_dataframe

df = clean_source_dataframe()

print(df.info())
print(df.isna().sum()/len(df))