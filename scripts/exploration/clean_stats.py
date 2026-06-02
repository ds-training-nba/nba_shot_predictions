from app.data_providers import provide_dataframe, DataFrameRequest

# Analyze NAs
request = DataFrameRequest(add_computed = False,
    filter_pre_encoding_columns = False,
    encode_for_model = False,
    filter_top_players=False,
                           filter_clean=False
)
df = provide_dataframe(request)
print(df[df['GAME_ID_x'].isna()])
print(df.info())
print(df['GAME_ID_x'].isna().sum()/len(df))