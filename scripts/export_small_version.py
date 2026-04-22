from app.data_providers import filtered_shots_dataframe

# get original shots filtered by players, then random sort and get 10000 first rows
df = filtered_shots_dataframe().sample(frac=1).head(10000)

df.to_csv('data/small/shots.csv')
