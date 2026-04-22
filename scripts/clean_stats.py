from app.data_providers import clean_source_dataframe, provide_dataframe, DataFrameRequest

request = DataFrameRequest(add_computed = False,
    filter_pre_encoding_columns = False,
    encode_for_model = False,
    filter_top_players=False
)
df = provide_dataframe(request)

print(df.info())
print(df.isna().sum()/len(df))