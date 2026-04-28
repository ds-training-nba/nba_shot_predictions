import pandas as pd
import pprint
from app.config import ESPN_TOP_25
file = 'data/orig/shots.csv'
df = pd.read_csv(file)

last_entries = {}
for player_name in ESPN_TOP_25:
    df_player = df[df['PLAYER_NAME']  == player_name]
    #print(player_name, ': Count ', len(df_player), ' last_entry ', df_player['Game Date'].max())
    last_entries[player_name] = df_player['Game Date'].max()

pprint.pp([ name for name, last_entry in sorted(last_entries.items(), key=lambda item: item[1])[5:]])

