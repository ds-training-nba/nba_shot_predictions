import pandas as pd
file = 'data/orig/shots.csv'
df = pd.read_csv(file)
print(df.head())